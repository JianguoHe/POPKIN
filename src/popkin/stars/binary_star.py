import numpy as np
from popkin.stars.single_star import SingleStar
from numba import float64, int64, types, from_dtype
from popkin.utils import conditional_jitclass
from popkin.constants import G, M_sun, R_sun, sec_per_year, day_per_year, sep_to_period, period_to_sep
from popkin.constants import struct_dtype_binary
from popkin.stars.mt_stability import MT_stability_MS
from popkin.stars.sn_binary import post_supernova_orbit
from popkin.config.controls_default import ini_spin_scheme, max_step, max_time, jit_enabled
from popkin.config.controls_default import alpha_CE, HG_survive_CE, M_ch, M_ECSN, M_ns_max
from popkin.config.controls_default import alpha_wind, beta_wind, mu_wind
from popkin.config.controls_default import eddfac, mass_accretion_model, epsnov, WD_crit_accretion, M_wd_ns_crit
from popkin.config.user_config import apply_user_config

apply_user_config(globals(), "inlist")


# Return the single-star class type (or class object) based on jitclass usage.
def get_star_instance_type():
    if jit_enabled:
        return SingleStar.class_type.instance_type
    else:
        return SingleStar


# Binary star class
spec = [
    ('star1', get_star_instance_type()),
    ('star2', get_star_instance_type()),
    ('totalmass', float64),                  # total mass of binary
    ('Z', float64),                          # initial metallicity
    ('ecc', float64),                        # eccentricity
    ('sep', float64),                        # orbital semimajor axis [unit: R_sun] (either sep or period can be provided)
    ('period', float64),                     # orbital period [unit: yr] (when provided as input, unit is day)
    ('omega', float64),                      # orbital angular frequency [unit: 1/yr]
    ('jorb', float64),                       # orbital angular momentum [unit: M_sun R_sun^2 / yr]
    ('dt', float64),                         # evolution timestep
    ('q1', float64),                         # mass ratio: m1/m2
    ('q2', float64),                         # mass ratio: m2/m1
    ('jdot', float64),                       # total rate of change of orbital angular momentum [unit: M_sun R_sun^2 / yr^2]
    ('jdot_wind', float64),                  # rate of change of orbital angular momentum due to stellar wind [unit: M_sun R_sun^2 / yr^2]
    ('jdot_tide', float64),                  # rate of change of orbital angular momentum due to tides [unit: M_sun R_sun^2 / yr^2]
    ('jdot_mt', float64),                    # rate of change of orbital angular momentum due to non-conservative mass transfer [unit: M_sun R_sun^2 / yr^2]
    ('jdot_gr', float64),                    # rate of change of orbital angular momentum due to gravitational-wave radiation [unit: M_sun R_sun^2 / yr^2]
    ('edot', float64),                       # rate of change of orbital eccentricity
    ('edot_wind', float64),                  # rate of change of orbital eccentricity due to stellar wind
    ('edot_gr', float64),                    # rate of change of orbital eccentricity due to gravitational-wave radiation
    ('edot_tide', float64),                  # rate of change of orbital eccentricity due to tides
    ('event', types.string),                 # event that occurs during the evolution of the star, including CE, RLOF begin, RLOF end, merge, disrupt, or None
    ('state', types.string),                 # current state ['detached', 'semidetached', 'contact', 'disrupted']
    ('ktype', int64[:, :]),                  # stellar types of post-collision merger products
    ('time', float64),                       # evolutionary time [unit: yr]
    ('step', int64),                         # current iteration number
    ('data', from_dtype(struct_dtype_binary)[:]),  # Array for storing properties at each timestep
    ('v_offset', float64[:]),                # barycenter velocity offset due to SN
    ('v1_offset', float64[:]),               # velocity offset of star 1 after system disruption
    ('v2_offset', float64[:]),               # velocity offset of star 2 after system disruption
    ('index', int64),                        # binary index, used as random seed
]


@conditional_jitclass(spec)
class BinaryStar:
    def __init__(self, star1, star2, ecc=0, sep=0, period=0, index=0):
        self.star1 = star1
        self.star2 = star2
        self.totalmass = star1.mass + star2.mass
        self.Z = star1.Z
        self.ecc = ecc
        self._set_orbital_parameter(sep, period)
        self.omega = 2 * np.pi / self.period
        self.jorb = self._set_jorb()
        self.dt = 0.
        self.jdot = 0.
        self.jdot_wind = 0.
        self.jdot_tide = 0.
        self.jdot_mt = 0.
        self.jdot_gr = 0.
        self.edot = 0.
        self.edot_wind = 0.
        self.edot_tide = 0.
        self.edot_gr = 0.
        self.event = 'None'
        self.state = 'detached'
        self.q1 = star1.mass / star2.mass
        self.q2 = star2.mass / star1.mass
        self.cal_radius_rochelobe()
        self.time = 0.
        self.step = 0
        self.data = np.zeros(max_step, dtype=struct_dtype_binary)
        self.v_offset = np.full(3, np.nan)
        self.v1_offset = np.full(3, np.nan)
        self.v2_offset = np.full(3, np.nan)
        self.index = index
        self._set_spin()
        self._set_ktype()
        np.random.seed(index)               # set random seed

    # ------------------------------------------------------------------------------------------------------------------
    #                                             Evolve the binary system.
    # ------------------------------------------------------------------------------------------------------------------
    def evolve(self):
        while self.step < max_step:
            # Check whether the binary has been disrupted; if so, evolve the two stars as single stars.
            # 检查双星是否瓦解, 瓦解则进入单星演化
            if self.state == 'disrupted':
                if self.time < max_time * 1e6:
                    self.evolve_disrupted_system()
                    continue
                else:
                    self.finish()
                    break
            
            # Update stellar mass, spin, surface temperature, thermal timescale, nuclear timescale, and other properties.
            # 更新恒星质量/自旋/表面温度/热力学时标/核时标, 以及各种参数
            self.star1.update()
            self.star2.update()

            # Check whether the binary evolves normally; 
            # if an exceptional event such as a Type Ia SN occurs, evolve it as a disrupted system.
            # 检查双星是否正常演化, 存在异常(Ia SN)则进入瓦解系统演化
            if self.star1.event == 'Ia' or self.star2.event == 'Ia':
                self.process_Ia_SN()
                continue

            # Update orbital angular momentum, eccentricity, semi-major axis, period, and angular frequency.
            # 更新轨道角动量/偏心率/半长轴/周期/角频率
            self.update()

            # Check whether the binary is disrupted by a supernova; if so, evolve it as a disrupted system.
            # 检查双星系统是否因为SN瓦解, 符合则进入瓦解系统演化
            if self.state == 'disrupted':
                continue

            # If a supernova occurs but the binary remains bound.
            # 若发生超新星爆炸, 同时双星系统没有瓦解
            if self.star1.event in {'AIC', 'ECSN', 'CCSN'} or self.star2.event in {'AIC', 'ECSN', 'CCSN'}:
                # If mass transfer was still occurring before the supernova.
                # 如果超新星爆炸前仍在向伴星转移物质
                if self.state == 'semidetached':
                    self.state = 'detached'
                    self.event = 'RLOF end'
                self.save()
                self.reset()
                self.update_time(use_min_timestep=True)
                self.update_step()
                continue

            # Reset variables.
            # 重置变量
            self.reset()

            # Include magnetic braking in the binary, reducing stellar spin angular momentum.
            # 考虑双星的磁制动影响（自旋角动量的减少）
            self.star1.magnetic_braking()
            self.star2.magnetic_braking()

            # Include stellar-wind effects: mass, spin angular momentum, and orbital angular momentum loss/gain.
            # 考虑星风的影响（质量/自旋角动量/轨道角动量的减少/增加）
            self.stellar_wind()

            # Include gravitational-wave radiation, reducing orbital angular momentum.
            # 考虑引力波辐射的影响(轨道角动量的减少)
            self.GW_radiation()

            # Include tidal circularization, orbital shrinkage, and spin evolution.
            # 考虑潮汐的圆化、轨道收缩和自旋
            self.tide_effect()

            # Refresh variables: total orbital-angular-momentum/eccentricity derivatives and 
            # total stellar mass/spin-angular-momentum derivatives.
            # 刷新变量(总的轨道角动量/偏心率变化率、总的恒星质量/自旋角动量变化率)
            self.refresh()

            # Check for Roche-lobe overflow.
            # 检查洛希瓣渗溢情况
            self.check_overfill()

            # A Type Ia SN occurs when a He WD accretes helium-rich material up to 0.7 Msun,
            # or when a CO WD accretes more than 0.15 Msun of helium-rich material.
            # 当氦白矮星吸积富氦物质达到0.7M_sun、碳氧白矮星吸积超过0.15M_sun富氦物质，都会发生Ia SN
            if self.star1.event == 'Ia' or self.star2.event == 'Ia':
                self.process_Ia_SN()
                continue
            
            # If CE evolution leads to a merger because orbital energy cannot eject the envelope,
            # or if a post-CE supernova disrupts the binary, evolve the system as disrupted.
            # 若发生公共包层演化且轨道能不足以驱散公共包层导致并合, 或公共包层后发生超新星爆炸导致双星瓦解, 都进入瓦解系统演化
            if self.state == 'disrupted':
                continue
            
            # If CE evolution occurs and the binary survives without supernova disruption.
            # 若发生公共包层演化且成功存活, 同时没有被超新星瓦解
            if self.event == 'CE':
                self.event = 'None'
                self.save()
                self.reset()
                self.update_time(use_min_timestep=True)
                self.update_step()
                continue
            
            # Determine the next timestep for each star from its current evolutionary phase (yr).
            # 通过两颗恒星的当前阶段确定各自下一步的步长(yr)
            self.star1.timestep()
            self.star2.timestep()

            # Semidetached binary.
            # 半接双星
            if self.state == 'semidetached':
                # The binary fills its Roche lobe already at ZAMS and undergoes stable mass transfer
                # (almost all such systems merge; a few outside the fitting range may not),
                # or it fills its Roche lobe for the first time after evolution.
                # ZAMS时就充满洛希瓣(基本都会并合, 少数拟合范围外的情况不会)且发生稳定物质转移
                # 或者演化后第一次充满洛希瓣
                if self.step == 0 or (self.step > 0 and self.data[self.step - 1]['state'] == b'detached'):
                    self.event = 'RLOF begin'
                    self.dt = self.dt * 0.001 if self.dt != 0 else 1
                    self.dt = min(self.dt, self.star1.dt, self.star2.dt)
                # During RLOF after the first contact, double the timestep while limiting the mass change
                # of the binary to less than 0.5% within one step.
                # 发生RLOF, 非第一次充满洛希瓣, 步长倍增, 同时限制双星在一个步长内质量变化不超过0.5%
                if self.step > 0 and self.data[self.step - 1]['state'] == b'semidetached':
                    self.star1.dt = min(self.star1.dt,
                                        0.005 * self.star1.mass / abs(self.star1.mdot_mt + self.star1.mdot_wind))
                    self.star2.dt = min(self.star2.dt,
                                        0.005 * self.star2.mass / abs(self.star2.mdot_mt + self.star2.mdot_wind))
                    self.dt = min(2 * self.dt, self.star1.dt, self.star2.dt)
            # Detached binary.
            # 分离双星
            else:
                if self.step > 0 and self.data[self.step - 1]['state'] == b'semidetached':
                    self.event = 'RLOF end'
                # For non-compact stars, limit mass loss to less than 1% and not more than the envelope mass.
                # 对于非致密星, 限制质量损失(<1%)且不超过包层质量
                self.star1.limit_mass_change()
                self.star2.limit_mass_change()
                # Limit orbital-angular-momentum change to less than 0.2%, and determine the next timestep
                # together with the two stellar evolutionary timesteps.
                # 限制轨道角动量变化(<0.2%), 同时根据另外两颗恒星的演化步长确定下一步步长
                self.jdot = self.jdot_wind + self.jdot_gr + self.jdot_tide + self.jdot_mt
                self.dt = min(0.002 * self.jorb / abs(self.jdot), self.star1.dt, self.star2.dt)
                # For stars close to Roche-lobe filling, keep the next overfilling factor below 1.002.
                # 对于快充满洛希瓣的恒星, 要控制下一次充满程度到1.002以内, 我暂时没想到更好的办法, 回溯数据太麻烦, 只能在这里控制一下步长
                star1_R_to_RL = self.star1.R_mt / self.star1.R_rl
                star2_R_to_RL = self.star2.R_mt / self.star2.R_rl
                if star1_R_to_RL > 0.9 and star1_R_to_RL >= star2_R_to_RL:
                    need_R_rl = 1 - star1_R_to_RL
                    delta_R_rl = star1_R_to_RL - self.data[self.step - 1]['R1_div_RL1']
                    if delta_R_rl > 0:
                        self.dt = self.dt * min(1, need_R_rl / delta_R_rl)
                elif star2_R_to_RL > 0.9 and star2_R_to_RL >= star1_R_to_RL:
                    need_R_rl = 1 - star2_R_to_RL
                    delta_R_rl = star2_R_to_RL - self.data[self.step - 1]['R2_div_RL2']
                    if delta_R_rl > 0:
                        self.dt = self.dt * min(1, need_R_rl / delta_R_rl)

            # Do not exceed the maximum evolution time.
            # 不超过最长演化时间
            if self.time < max_time * 1e6:
                self.dt = min(self.dt, max_time * 1e6 - self.time, ((self.time // 1e9) + 1) * 1e9 - self.time)
            # If the maximum evolution time has been reached, finish the evolution.
            # 如果达到最长演化时间, 结束演化
            else:
                self.finish()
                break
            
            # Readjust the spin-change rate caused by tidal synchronization to avoid excessive transfer
            # between spin and orbital angular momentum.
            # 重新调整潮汐同步导致的恒星自旋变化率，防止自旋/轨道角动量之间的过度转移
            self.tide_effect(adjustment=True)

            # Refresh variables again: total orbital-angular-momentum/eccentricity derivatives
            # and total stellar mass/spin-angular-momentum derivatives.
            # 再次刷新变量(总的轨道角动量/偏心率变化率、总的恒星质量/自旋角动量变化率)
            self.refresh()

            # Save the current binary properties.
            # 保存双星的当前属性
            self.save()

            # Update the evolution time and stellar ages for the next step.
            # 更新下一步的演化时间和恒星年龄
            self.update_time()

            # Update the iteration counter.
            # 更新迭代次数
            self.update_step()

    # ------------------------------------------------------------------------------------------------------------------
    #                                        Update evolution time and stellar ages.
    #                                                  更新演化时间和恒星年龄
    # ------------------------------------------------------------------------------------------------------------------
    def update_time(self, use_min_timestep=False):
        # Use the minimum timestep after special events such as CE, merger, collision, or disruption.
        # 对于特殊事件(CE/merge/collision/disrupt)后的演化设置最小步长
        if use_min_timestep:
            self.dt = 1.0

        self.star1.dt = self.dt
        self.star2.dt = self.dt
        self.time = self.time + self.dt
        self.star1.time = self.star2.time = self.time
        self.star1.age = self.star1.age + self.dt / 1e6
        self.star2.age = self.star2.age + self.dt / 1e6

    # ------------------------------------------------------------------------------------------------------------------
    #                                            Update the iteration counter.
    #                                                      更新迭代次数
    # ------------------------------------------------------------------------------------------------------------------
    def update_step(self):
        self.step = self.step + 1
        self.star1.step = self.step
        self.star2.step = self.step

    # ------------------------------------------------------------------------------------------------------------------
    #                                                 Finish the evolution.
    # ------------------------------------------------------------------------------------------------------------------
    def finish(self):
        # In some cases, such as disruption by a supernova at the final moment,
        # do not record the final system state repeatedly.
        # 在某些情况(最后一刻发生SN瓦解), 不再重复记录系统最终状态
        if self.data[self.step - 1]['time'] >= max_time:
            end_step = self.step
        else:
            self.save()
            end_step = self.step + 1

        self.data = self.data[:end_step]
        self.star1.data = self.star1.data[:end_step]
        self.star2.data = self.star2.data[:end_step]

    # ------------------------------------------------------------------------------------------------------------------
    #                                             Save the current properties.
    #                                                    保存当前属性
    # ------------------------------------------------------------------------------------------------------------------
    def save(self):
        self.star1.save()
        self.star2.save()
        self.data[self.step]['time'] = self.time / 1e6
        self.data[self.step]['ecc'] = self.ecc
        self.data[self.step]['period'] = self.period * day_per_year
        self.data[self.step]['sep'] = self.sep
        self.data[self.step]['type1'] = self.star1.type
        self.data[self.step]['type2'] = self.star2.type
        self.data[self.step]['m1'] = self.star1.mass
        self.data[self.step]['m2'] = self.star2.mass
        self.data[self.step]['mc1'] = self.star1.M_core
        self.data[self.step]['mc2'] = self.star2.M_core
        self.data[self.step]['R1_div_RL1'] = self.star1.R_mt / self.star1.R_rl if self.state != 'disrupted' else -1
        self.data[self.step]['R2_div_RL2'] = self.star2.R_mt / self.star2.R_rl if self.state != 'disrupted' else -1
        self.data[self.step]['event'] = self.event_map()
        self.data[self.step]['state'] = self.state_map()
        self.data[self.step]['jorb'] = self.jorb
        self.data[self.step]['jdot'] = self.jdot
        self.data[self.step]['jdot_wind'] = self.jdot_wind
        self.data[self.step]['jdot_tide'] = self.jdot_tide
        self.data[self.step]['jdot_mt'] = self.jdot_mt
        self.data[self.step]['jdot_gr'] = self.jdot_gr
        self.data[self.step]['edot'] = self.edot
        self.data[self.step]['edot_wind'] = self.edot_wind
        self.data[self.step]['edot_tide'] = self.edot_tide
        self.data[self.step]['edot_gr'] = self.edot_gr
        self.data[self.step]['v_offset_x'] = self.v_offset[0]
        self.data[self.step]['v_offset_y'] = self.v_offset[1]
        self.data[self.step]['v_offset_z'] = self.v_offset[2]
        self.data[self.step]['v1_offset_x'] = self.v1_offset[0]
        self.data[self.step]['v1_offset_y'] = self.v1_offset[1]
        self.data[self.step]['v1_offset_z'] = self.v1_offset[2]
        self.data[self.step]['v2_offset_x'] = self.v2_offset[0]
        self.data[self.step]['v2_offset_y'] = self.v2_offset[1]
        self.data[self.step]['v2_offset_z'] = self.v2_offset[2]

    # ------------------------------------------------------------------------------------------------------------------
    #                                        Update the current orbital parameters.
    #                                                   更新当前轨道参数
    # ------------------------------------------------------------------------------------------------------------------
    def update(self):
        # Refresh variables: total orbital-angular-momentum and eccentricity derivatives.
        # 刷新变量(总的轨道角动量/偏心率变化率)
        self.refresh()

        # Update the total binary mass.
        # 更新双星总质量
        self.totalmass = self.star1.mass + self.star2.mass

        # Update orbital angular momentum.
        # 更新轨道角动量
        self.jorb += self.jdot * self.dt

        # Update orbital eccentricity.
        # 更新轨道偏心率
        self.ecc += self.edot * self.dt
        self.ecc = 0 if self.ecc < 1e-10 else self.ecc

        # If a supernova occurred, include the effect of the natal kick on the orbit.
        # 如果发生超新星爆炸, 考虑natal kick对轨道的影响
        self.check_SN()

        if self.state == 'disrupted':
            return
        else:
            # Update semi-major axis, period, and angular frequency.
            # 更新轨道半长轴/周期/角频率
            self.sep = self.totalmass * self.jorb ** 2 / (
                    (self.star1.mass * self.star2.mass * 2 * np.pi) ** 2 * period_to_sep ** 3 * (1 - self.ecc ** 2))
            self.period = sep_to_period * (self.sep ** 3 / self.totalmass) ** 0.5
            self.omega = 2 * np.pi / self.period

            # Update Roche-lobe radii.
            # 更新洛希瓣大小
            self.cal_radius_rochelobe()

    # ------------------------------------------------------------------------------------------------------------------
    #                        Handle the orbital and stellar properties after binary disruption.
    #                                                 双星瓦解后的参数处理
    # ------------------------------------------------------------------------------------------------------------------
    def process_disrupted_system(self):
        self.state = 'disrupted'
        self.ecc = -1
        self.period = 0
        self.sep = 0
        self.jorb = 0
        self.star1.R_rl = -1
        self.star2.R_rl = -1
        self.save()
        self.reset()
        self.state = 'disrupted'  # 由于reset会重置状态, 这里需要二次更新
        self.update_time(use_min_timestep=True)
        self.update_step()

    # ------------------------------------------------------------------------------------------------------------------
    #                                          Handle Type Ia supernova disruption.
    #                                                  处理 Ia SN 的情况
    # ------------------------------------------------------------------------------------------------------------------
    def process_Ia_SN(self):
        self.event = 'disrupt'
        if self.star1.event == 'Ia':
            self.supernova_in_binary(self.star1, self.star2)
        else:
            self.supernova_in_binary(self.star2, self.star1)
        self.process_disrupted_system()

    # ------------------------------------------------------------------------------------------------------------------
    #                          Continue the evolution after the binary orbit has been disrupted.
    #                                                  双星轨道瓦解后续演化
    # ------------------------------------------------------------------------------------------------------------------
    def evolve_disrupted_system(self):
        # Only one stellar component remains.
        # 只剩下单星
        if self.star1.type != 15 and self.star2.type == 15:
            self.star1.evolve(loop=False)
            self.star2.StellarProp()
            self.dt = self.star1.dt
            self.star1.age = self.star1.age + self.dt / 1e6
        elif self.star1.type == 15 and self.star2.type != 15:
            self.star2.evolve(loop=False)
            self.star1.StellarProp()
            self.dt = self.star2.dt
            self.star2.age = self.star2.age + self.dt / 1e6
        # Both stars survive, but the binary orbit has been disrupted.
        # 双星都存在只是轨道瓦解
        elif self.star1.type != 15 and self.star2.type != 15:
            self.star1.evolve(loop=False)
            self.star2.evolve(loop=False)
            self.star1.dt = self.star2.dt = self.dt = min(self.star1.dt, self.star2.dt)
            self.star1.age = self.star1.age + self.dt / 1e6
            self.star2.age = self.star2.age + self.dt / 1e6
        # Neither component remains; this is rare and mainly possible for double-degenerate Type Ia SN models.
        # 双星都消失了(通常不会发生, 除非是双简并模型Ia SN), 直接一步到位
        else:
            self.dt = max_time * 1e6 - self.time

        # Add velocity offsets if kicks occurred.
        # 如果发生kick, 则添加速度偏移
        self.v1_offset = self.star1.v_kick
        self.v2_offset = self.star2.v_kick

        # Save the current binary state.
        # 保存双星的当前属性
        self.save()

        # For stars that just underwent a supernova, set all mass-loss/transfer rates to zero to prevent further evolution.
        # 对于发生超新星爆炸的恒星, 将其质量变化率为零, 防止继续演化
        for star in [self.star1, self.star2]:
            if star.event in {'AIC', 'ECSN', 'CCSN', 'Ia'}:
                star.mdot = star.mdot_wind = star.mdot_wind_loss = star.mdot_wind_acc = star.mdot_mt = 0

        # Clear transient event and kick-velocity records.
        # 重置各类速度
        self.star1.event = 'None'
        self.star2.event = 'None'
        self.star1.v_kick = np.full(3, np.nan)
        self.star2.v_kick = np.full(3, np.nan)
        self.v1_offset = np.full(3, np.nan)
        self.v2_offset = np.full(3, np.nan)

        # Advance the evolution time and stellar ages to the next step.
        # 更新下一步的演化时间和恒星年龄
        self.star1.time = self.star2.time = self.time = self.time + self.dt

        # Update the iteration counter.
        # 更新迭代次数
        self.update_step()

    # ------------------------------------------------------------------------------------------------------------------
    #                                           Refresh the derivative terms.
    #                                                       刷新变量
    # ------------------------------------------------------------------------------------------------------------------
    def refresh(self):
        self.star1.refresh()
        self.star2.refresh()
        self.jdot = self.jdot_wind + self.jdot_tide + self.jdot_mt + self.jdot_gr
        self.edot = self.edot_wind + self.edot_tide + self.edot_gr

    # ------------------------------------------------------------------------------------------------------------------
    #                                   Reset the derivative terms and transient flags.
    #                                                       重置变量
    # ------------------------------------------------------------------------------------------------------------------
    def reset(self):
        self.star1.reset()
        self.star2.reset()
        self.jdot = self.jdot_wind = self.jdot_tide = self.jdot_mt = self.jdot_gr = 0.
        self.edot = self.edot_wind = self.edot_tide = self.edot_gr = 0.
        # Initialize the current binary state and event flag.
        # 初始化当前状态和事件
        self.state = 'detached'
        self.event = 'None'
        # Reset velocity offsets.
        # 重置速度偏移
        self.v_offset = np.full(3, np.nan)
        self.v1_offset = np.full(3, np.nan)
        self.v2_offset = np.full(3, np.nan)

    # ------------------------------------------------------------------------------------------------------------------
    #                               Check whether either component overfills its Roche lobe.
    #                                                    检查充满洛希瓣情况
    # ------------------------------------------------------------------------------------------------------------------
    def check_overfill(self):
        stars = [self.star1, self.star2]
        q = [self.q1, self.q2]
        # Use the ratio between the stellar mass-transfer radius and the periastron Roche-lobe radius.
        # 考虑恒星半径和对应的近心点洛希瓣半径的比值
        r_rl = [self.star1.R_mt / (self.star1.R_rl * (1 - self.ecc)),
                self.star2.R_mt / (self.star2.R_rl * (1 - self.ecc))]

        # Neither component overfills its Roche lobe.
        # 双星都未充满洛希瓣
        if r_rl[0] < 1 and r_rl[1] < 1:
            return

        # Choose the component with the deeper Roche-lobe overflow.
        # 确定充满洛希瓣程度更深的恒星
        i = r_rl.index(max(r_rl))

        # Determine whether mass transfer is dynamically stable.
        # 确定物质转移的稳定性
        if r_rl[0] >= 1 and r_rl[1] >= 1:
            # If both stars overfill their Roche lobes, the interaction is assumed to be unstable.
            # 双星都充满洛希瓣, 一定不稳定
            stable = False
            # if stars[i].type >= 10:
            #     self.merge(i)
            # else:
            #     self.CE_evolution(i)
        else:
            # NS/BH binary: merge directly.
            # 中子星/黑洞双星 → 直接合并
            if stars[i].type in {13, 14} and stars[1 - i].type in {13, 14}:
                stable = False
            # WD + NS/BH system: form an UCXB or merge depending on the WD mass.
            # 白矮星+中子星/黑洞 → UCXBs/合并
            elif stars[i].type in {10, 11, 12} and stars[1 - i].type in {13, 14}:
                if stars[i].mass > M_wd_ns_crit:
                    stable = False
                else:
                    stable = True
            # Double-WD system: AM CVn channel or merger.
            # 双白矮星 → AM CVn/合并
            elif stars[i].type in {10, 11, 12} and stars[1 - i].type in {10, 11, 12}:
                if stars[i].mass / stars[1 - i].mass > 0.628:
                    stable = False
                else:
                    stable = True
            # Two hydrogen-rich stars.
            # 两个富氢恒星
            elif (stars[i].type in {0, 1, 2} or (stars[i].type == 4 and stars[i].mass0 >= 12)) and stars[
                1 - i].type <= 2:
                # ZAMS 充满洛希瓣
                if self.step == 0:
                    mass1i = self.star1.mass
                    mass2i = self.star2.mass
                    tbi = self.period * day_per_year
                # ZAMS 未充满洛希瓣
                else:
                    mass1i = self.data[0]['m1']
                    mass2i = self.data[0]['m2']
                    tbi = self.data[0]['period']
                # print(i, self.step, mass1i, mass2i, tbi)
                if i == 0:
                    qc = MT_stability_MS(stars[i].type, mass1i, mass2i, tbi, mass_accretion_model)
                else:
                    qc = MT_stability_MS(stars[i].type, mass2i, mass1i, tbi, mass_accretion_model)
                # qc = MT_stability_MS(stars[i].type, stars[i].mass0, stars[1 - i].mass0, self.data[0]['period'],
                #                      mass_accretion_model=mass_accretion_model)
                if q[i] > qc:
                    stable = False
                else:
                    stable = True
            # Stability criterion for mass transfer from a hydrogen-rich donor to an NS/BH accretor.
            # 中子星/黑洞 + 富氢恒星的物质转移稳定性判据
            elif stars[i].type <= 6 and stars[1 - i].type in {13, 14}:
                # 【Shao, Y., & Li, X.-D. 2021, ApJ, 920, 81】
                if q[i] < 2:
                    stable = True
                elif q[i] > 2.1 + 0.8 * stars[1 - i].mass:
                    stable = False
                else:
                    radmax = 10 ** (-0.5583 * np.log10(stars[i].mass) ** 2 + 2.6937 * np.log10(stars[i].mass) + 0.2573)
                    radmin = 6.6 - 26.1 * q[i] + 11.4 * q[i] ** 2
                    if stars[i].R > radmax or stars[i].R < radmin:
                        stable = False
                    else:
                        stable = True
                # zeta_Adiabatic from eq (57/61) Soberman, Phinney, vdHeuvel (1997)
                # zeta_RL from Woods et al., 2012
                # if stars[i].type >= 2:
                #     mc = stars[i].M_core / stars[i].mass
                #     term = 2 / 3 * mc / (1 - mc) - 1 / 3 * (1 - mc) / (1 + 2 * mc)
                #     zeta_Adiabatic = term - 0.03 * mc + 0.2 * mc / (1 + 1 / (1 - mc) ** 6)
                #     beta = 1
                #     term1 = 2 * (q[i] - 1) - q[i] * (1 - beta) / (q[i] + 1)
                #     term2 = q[i] ** (1/3) * (1.2 * q[i] ** (1/3) + 1 / (1 + q[i] ** (1/3)))
                #     term3 = 3 * (0.6 * q[i] ** (2/3) + np.log(1 + q[i] ** (1/3)))
                #     term4 = 1 + beta * q[i]
                #     zeta_RL = term1 + (2 / 3 - term2 / term3) * term4
                #     if zeta_Adiabatic < zeta_RL:
                #         self.CE_evolution(i)
                #     else:
                #         self.RLOF(i)
            # Hertzsprung-gap donor.
            # 赫氏空隙作为donor星
            elif stars[i].type == 2:
                qc = 4
                if q[i] > qc:
                    stable = False
                else:
                    stable = True
            # Giant donor.
            # 巨星作为donor星
            elif stars[i].type in {3, 5, 6}:
                # qc = (1.67d0-zpars(7)+2.d0*(massc(j1)/mass(j1))**5)/2.13d0
                # Alternatively use condition of Hjellming & Webbink, 1987, ApJ, 318, 794.
                qc = 0.362 + 1 / (3 * (1 - stars[i].M_core / stars[i].mass))
                if q[i] > qc:
                    stable = False
                else:
                    stable = True
            # Hydrogen main-sequence/helium-star donor with a helium-star companion.
            # 氢主序/氦星 + 氦星
            elif stars[i].type in {0, 1, 7, 8, 9} and stars[1 - i].type in {7, 8, 9}:
                stable = True
            # Helium-star donor with a compact companion.
            # 氦星 + 致密星
            elif stars[i].type in {7, 8, 9} and stars[1 - i].type in {13, 14}:
                # If the period is shorter than 0.06 d, CE evolution may occur (Tauris 2015, MNRAS, 451, 2123).
                # 如果周期小于0.06天, 则可能会发生CE (Tauris, T. 2015, MNRAS, 451, 2123)
                if stars[i].mass > 2.7 and self.period <= 1.644e-4:
                    qc = 0.01
                else:
                    qc = 10
                if q[i] > qc:
                    stable = False
                else:
                    stable = True
            else:
                qc = 3.0
                if q[i] > qc:
                    stable = False
                else:
                    stable = True

            # If the donor overfills its Roche lobe very deeply, the orbit is extremely tight,
            # and the companion is compact, gravitational radiation can become so strong that
            # the orbital energy would become negative; in that case force a merger.
            # This differs from the traditional BSE treatment: for BH + giant semidetached systems,
            # the stability is judged using Shao & Li (2021), regardless of the giant's overfilling factor,
            # because the BSE radius is a fitting radius rather than the physical radius.
            # 如果是donor星充满洛希瓣程度太深, 双星轨道间距太近, 伴星也是致密星, 引力波会非常强烈, 可能会产生负的轨道能, 这时候让它们并合
            # 这里有个地方和传统bse处理方式不一样, 对于黑洞+巨星半接系统, 根据shao2021的结果判定稳定性,
            # 不管巨星充满洛希瓣的渗溢程度, 因为bse中半径是拟合半径, 而非真实半径
            if r_rl[i] > 7 and self.period <= 2.7e-4 and stars[1 - i].type >= 10:
                stable = False

            # For highly eccentric systems, at least one component must be an NS/BH, so treat the interaction as a collision.
            # 如果系统是极端偏心系统, 则必有一个致密星为中子星/黑洞, 进入碰撞演化
            if self.ecc > 0.5:
                stable = False

        # Route the system to stable RLOF, direct collision, or CE evolution according to the component types.
        # 根据双星类型进入RLOF/碰撞/CE演化
        if stable:
            self.RLOF(i)
        else:
            types_without_core = {0, 1, 7, 10, 11, 12, 13, 14}
            if stars[i].type in types_without_core and stars[1 - i].type in types_without_core:
                self.collision(i)
            else:
                self.CE_evolution(i)

    # ------------------------------------------------------------------------------------------------------------------
    #                                      Dynamically stable Roche-lobe overflow
    #                                                  动力学稳定物质转移
    # ------------------------------------------------------------------------------------------------------------------
    def RLOF(self, i):
        # Set the binary to a semidetached state.
        # 更新双星状态
        self.state = 'semidetached'

        # Store the two stars in a list for compact indexing.
        # 添加恒星列表, 方便调用
        stars = [self.star1, self.star2]
        q = [self.q1, self.q2]
        r_rl = [self.star1.R_mt / self.star1.R_rl, self.star2.R_mt / self.star2.R_rl]

        # Nuclear-timescale mass-transfer rate in units of Msun/yr.
        # 核时标物质转移(太阳质量/年)
        stars[i].mdot_mt = -3e-6 * np.log(r_rl[i]) ** 3 * min(stars[i].mass, 5) ** 2

        # Reduce the mass-transfer rate for Hertzsprung-gap donors.
        # The physical reason for this correction is inherited from the BSE treatment and is not fully clear here.
        # 降低HG恒星的物质转移速率, 俺也不知道为什么
        if stars[i].type == 2:
            stars[i].mdot_mt = stars[i].mdot_mt * max(1 - stars[i].M_core / stars[i].mass, 0.01)

        # For WD donors, the initial mass-transfer rate can be very large and roughly scales with mass.
        # Therefore, in WD-to-NS systems the early mass-transfer phase is short: the WD mass can quickly drop
        # below 0.1 Msun, followed by a long-lived low-rate transfer phase that may last up to a Hubble time.
        # 对致密星WD来说, 物质转移速率会很大, 通常与质量成正比, 因此WD向NS转移物质的初期时标很短, 很快WD的质量就下降到0.1M_sun以下,
        # 之后就是长期的低速率物质转移, 甚至可以到哈勃时标结束
        if stars[i].type >= 10:
            stars[i].mdot_mt = stars[i].mdot_mt * 1e3 * stars[i].mass / max(stars[i].R, 1e-4)

        # For giant-like donors, limit the transfer rate by the thermal timescale.
        # For other stellar types, limit it by the dynamical timescale.
        # 对于类巨星, 限制物质转移速率在热速率内, 对于其他类型的恒星, 限制在动力学速率内
        if 2 <= stars[i].type <= 9 and stars[i].type != 7:
            stars[i].mdot_mt = -min(abs(stars[i].mdot_mt), stars[i].mass / stars[i].tau_kh)
        else:
            stars[i].mdot_mt = -min(abs(stars[i].mdot_mt), stars[i].mass / stars[i].tau_dyn)

        # Mass-transfer timescale of the donor.
        # 计算donor星的物质转移时标
        tau_mt = stars[i].mass / abs(stars[i].mdot_mt)

        # Specific orbital angular momentum carried away by non-accreted material.
        # This assumes that material not retained by the accretor, e.g. nova ejecta or super-Eddington outflows,
        # leaves the system with the specific orbital angular momentum of the accretor.
        # 吸积星的比轨道角动量(假设没有吸积的物质[新星、超爱丁顿]会带走吸积星的比轨道角动量)
        specific_angular_momentum = self.omega * self.sep ** 2 * np.sqrt(1.0 - self.ecc ** 2) * stars[
            i].mass ** 2 / self.totalmass ** 2

        # Determine the accretion efficiency of the companion.
        # Accretion onto H-rich stars: several accretion-efficiency prescriptions are available.
        # 确定伴星的吸积效率
        # 对于富氢恒星的吸积, 考虑几种吸积效率模型
        if stars[1 - i].type in {0, 1, 2, 4}:
            if mass_accretion_model == 'rotation dependent':
                # Rotation-dependent accretion model.
                # 旋转依赖模型
                max_spin = 2 * np.pi * np.sqrt(stars[1 - i].mass * period_to_sep ** 3 / stars[1 - i].R ** 3)
                stars[1 - i].mdot_mt = -min(1, 1 - stars[1 - i].spin / max_spin) * stars[i].mdot_mt
            elif mass_accretion_model == 'half accretion':
                # Half-accretion model.
                # 一半吸积
                stars[1 - i].mdot_mt = - stars[i].mdot_mt * 0.5
            elif mass_accretion_model == 'thermal equilibrium':
                # Thermal-equilibrium-limited accretion model.
                # 热平衡限制模型
                stars[1 - i].mdot_mt = -min(1, 10 * tau_mt / stars[1 - i].tau_kh) * stars[i].mdot_mt
            else:
                raise ValueError(
                    "Unsupported mass_accretion_model. Expected one of: "
                    "'rotation dependent', 'half accretion', 'thermal equilibrium'."
                )

        elif stars[1 - i].type in {3, 5, 6}:
            # In principle, giant accretors should rarely occur here. If they do, the convective envelope
            # is assumed to retain all transferred material.
            # 理论上不应该出现这种巨星吸积的情况, 如果有的话, 对流包层可以吸积所有物质
            stars[1 - i].mdot_mt = - stars[i].mdot_mt

        elif stars[1 - i].type in {7, 8, 9}:
            # Naked helium star secondary swells up to a core helium burning star or SAGB star
            # unless the primary is also a helium star.
            if stars[i].type >= 7:
                stars[1 - i].mdot_mt = -min(1, 10 * tau_mt / stars[1 - i].tau_kh) * stars[i].mdot_mt
            # The helium star is engulfed by a hydrogen-rich envelope and becomes a CHeB/TPAGB star.
            # 氦星吞没在氢包层中成为CHeB/TPAGB star
            else:
                stars[1 - i].mdot_mt = - stars[i].mdot_mt

                # If the accretion rate exceeds the wind mass-loss rate and the mass accreted within one Keplerian orbit
                # is larger than 1e-4 of the stellar mass, treat the accretor as engulfed in the donor envelope
                # and initiate common-envelope evolution.
                # 如果吸积星的质量吸积率大于星风损失率, 且一个开普勒轨道周期内的质量吸积量大于1e-4恒星的质量, 
                # 则认为恒星被吞没在伴星的包层中, 进入公共包层演化
                if (stars[1 - i].mdot_mt + stars[1 - i].mdot_wind > 0. and 
                    stars[1 - i].mdot_mt * self.period > 1e-4 * stars[1 - i].mass):
                    # Evolve one timestep during which the transferred material builds up an envelope.
                    # The helium star then becomes a CHeB/TPAGB star through accretion of H-rich material.
                    # 考虑一个步长, 在此步长内转移过来的物质形成包层, 氦星通过吸积富氢物质变成CHeB/TPAGB
                    self.save()
                    self.dt = min(2 * self.dt, 0.005 * stars[i].mass / abs(stars[i].mdot_mt + stars[i].mdot_wind))
                    self.update_time()
                    self.update_step()
                    mass_gain = max(0.001, stars[1 - i].mdot_mt * self.dt)
                    stars[i].mass -= mass_gain

                    # HeMS → CHeB
                    if stars[1 - i].type == 7:
                        stars[1 - i].type = 4
                        stars[1 - i].M_core = stars[1 - i].mass
                        stars[1 - i].mass += mass_gain
                        age_frac = stars[1 - i].age / stars[1 - i].tm
                        stars[1 - i].solve_initial_mass_CHeB(stars[1 - i].M_core, age_frac)
                        stars[1 - i].StellarCal()
                        stars[1 - i].StellarCal()
                        if stars[1 - i].type == 3:
                            stars[1 - i].age = stars[1 - i].tscls[1] + 1e-6 * (stars[1 - i].tscls[2] - stars[1 - i].tscls[1])
                        else:
                            stars[1 - i].age = stars[1 - i].tscls[2] + age_frac * stars[1 - i].tscls[3]
                    # He-giant → TPAGB
                    else:
                        stars[1 - i].type = 6
                        stars[1 - i].mass += stars[1 - i].mdot_mt * self.dt
                        stars[1 - i].solve_initial_mass_TPAGB(stars[1 - i].M_core)
                        stars[1 - i].StellarCal()
                        stars[1 - i].age = stars[1 - i].tscls[13]

                    # Update the accretor type.
                    # 更新吸积星类型
                    self.save()
                    self.update_step()
                    self.update_time(use_min_timestep=True)

                    # Force the binary into common-envelope evolution.
                    # 让双星进入公共包层演化
                    stars[1 - i].StellarProp()
                    stars[1 - i].R_mt = stars[1 - i].R
                    self.CE_evolution(i)
                    return

        # White dwarf accreting H-rich material.
        # 白矮星吸积富氢物质
        elif stars[1 - i].type in {10, 11, 12} and stars[i].type <= 6:
            # Lower accretion-rate limit for stable hydrogen burning (Wang B. 2018, doi:10.1088/1674-4527/18/5/49, Eq. 2).
            # 稳定氢燃烧的最低吸积率
            M_dot_stable = 2.93e-7 * (
                    -stars[1 - i].mass ** 3 + 4.41 * stars[1 - i].mass ** 2 - 3.38 * stars[1 - i].mass + 0.84)
            # Upper critical accretion rate for stable hydrogen burning (Wang B. 2018, doi:10.1088/1674-4527/18/5/49, Eq. 1).
            # 稳定氢燃烧的最高临界吸积率
            M_dot_crit = 0.27e-7 * (stars[1 - i].mass ** 2 + 25.52 * stars[1 - i].mass - 9.02)

            # Accretion continues until a nova eruption; most of the accreted material is expelled,
            # while the WD retains only a small fraction (Hurley et al. 2002, Eq. 66).
            # 持续吸积直到新星爆发, 同时吹散大部分的吸积物质, 白矮星保留少量吸积物质
            if abs(stars[i].mdot_mt) < M_dot_stable:
                stars[1 - i].mdot_mt = - stars[i].mdot_mt * epsnov
            
            # Stable nuclear burning on the WD surface; this phase can appear as an X-ray source.
            # 在白矮星表面稳定燃烧(X射线源)
            elif M_dot_stable <= abs(stars[i].mdot_mt) <= M_dot_crit:
                stars[1 - i].mdot_mt = - stars[i].mdot_mt
            # Super-critical WD accretion. Available treatments include common-envelope evolution,
            # optically thick wind, and common-envelope wind.
            # 白矮星超临界吸积, 可选用模型: 公共包层、光学厚星风、CE星风
            else:
                # Specific orbital angular momentum carried away by the common-envelope wind
                # (Cui et al. 2022, doi:10.1051/0004-6361/202141335, Eq. 11).
                # 公共包层星风模型带走的比轨道角动量
                if WD_crit_accretion == 'CE-wind':
                    stars[1 - i].mdot_mt = M_dot_crit
                    specific_angular_momentum = self.omega * (self.sep + 0.1 * self.sep) ** 2
                # Optically thick wind model.
                # 光学厚星风模型
                elif WD_crit_accretion == 'OTW':
                    stars[1 - i].mdot_mt = M_dot_crit
                # Common-envelope model.
                # 公共包层模型
                elif WD_crit_accretion == 'CE':
                    stars[1 - i].mdot_mt = - stars[i].mdot_mt
                    self.save()

                    # Evolve one timestep during which the WD accretes H-rich material and develops a giant-like envelope.
                    # 考虑一个步长, 在此步长内白矮星通过吸积富氢物质变成巨星
                    self.dt = min(2 * self.dt, 0.005 * stars[i].mass / abs(stars[i].mdot_mt + stars[i].mdot_wind))
                    self.update_time()
                    self.update_step()
                    mass_gain = max(0.001, stars[1 - i].mdot_mt * self.dt)

                    # The transferred material forms an envelope within one timestep.
                    # 在一个步长内转移过来的物质形成包层
                    stars[i].mass -= mass_gain
                    stars[1 - i].mass += mass_gain

                    # HeWD → GB
                    if stars[1 - i].type == 10:
                        stars[1 - i].type = 3
                        stars[1 - i].solve_initial_mass_GB(stars[1 - i].M_core)
                        stars[1 - i].StellarCal()
                        stars[1 - i].age = stars[1 - i].tscls[1] + 1e-6 * (
                                    stars[1 - i].tscls[2] - stars[1 - i].tscls[1])
                    # COWD/ONeWD → TPAGB
                    else:
                        stars[1 - i].type = 6
                        stars[1 - i].solve_initial_mass_TPAGB(stars[1 - i].M_core)
                        stars[1 - i].StellarCal()
                        stars[1 - i].age = stars[1 - i].tscls[13]

                    # Force the binary into common-envelope evolution.
                    # 让双星进入公共包层演化
                    stars[1 - i].StellarProp()
                    stars[1 - i].R_mt = stars[1 - i].R
                    self.CE_evolution(i)
                    return
                else:
                    raise ValueError(
                        "Unsupported WD_crit_accretion. Expected one of: 'CE-wind', 'OTW', 'CE'."
                    )

        # He WD accreting He-rich material.
        # 氦白矮星吸积富氦物质
        elif stars[1 - i].type == 10 and stars[i].type in {7, 8, 9, 10}:
            stars[1 - i].mdot_mt = - stars[i].mdot_mt * 0.5
            # A He WD can grow by accreting He-rich material only up to 0.7 Msun;
            # above this limit it is assumed to undergo a supernova.
            # 氦白矮星只能吸积富氦物质达到0.7M_sun, 否则就会发生超新星
            if stars[1 - i].mass > 0.7:
                stars[1 - i].type = 15
                stars[1 - i].mass = 0
                stars[1 - i].event = 'Ia'
                return

        # CO WD accreting He/CO-rich material.
        # 碳氧白矮星吸积富He/CO物质
        elif stars[1 - i].type == 11 and stars[i].type in {7, 8, 9, 10, 11, 12}:
            stars[1 - i].mdot_mt = - stars[i].mdot_mt * 0.5
            # A CO WD can retain at most 0.15 Msun of He-rich material;
            # above this limit it is assumed to undergo a Type Ia supernova.
            # 碳氧白矮星只能吸积最多0.15M_sun的富氦物质, 否则就会发生Ia超新星
            if stars[i].type < 11 and stars[1 - i].mass - stars[1 - i].mass0 > 0.15:
                stars[1 - i].type = 15
                stars[1 - i].mass = 0
                stars[1 - i].event = 'Ia'
                return

        # ONe WD accreting He/CO-rich material.
        # 氧氖白矮星吸积富He/CO物质
        elif stars[1 - i].type == 12 and stars[i].type in {7, 8, 9, 10, 11, 12}:
            stars[1 - i].mdot_mt = - stars[i].mdot_mt * 0.5
            # If the ONe WD exceeds M_ECSN, it will undergo AIC.
            # This is not checked here explicitly, but is handled automatically through the single-star properties.
            # 氧氖白矮星如果质量超过M_ECSN, 会经历AIC, 但不在这检查, 而是通过单星类的属性自动判定

        # The accretor is a neutron star or black hole.
        # 吸积星为中子星/黑洞
        else:
            # Half of the transferred material is accreted.
            # 一半吸积
            stars[1 - i].mdot_mt = - stars[i].mdot_mt * 0.5

        # Limit all accretion to the Eddington accretion rate.
        # 将所有的吸积限定在爱丁顿吸积率
        edd_limit = 2.08e-3 * eddfac * (1 / (1 + stars[1 - i].zpars[11])) * stars[1 - i].R
        stars[1 - i].mdot_mt = min(stars[1 - i].mdot_mt, edd_limit)

        # Orbital angular-momentum loss caused by non-conservative mass transfer.
        # 考虑由于物质转移不守恒导致的轨道角动量变化率
        self.jdot_mt = (stars[i].mdot_mt + stars[1 - i].mdot_mt) * specific_angular_momentum

        # Spin angular-momentum change of the donor caused by mass transfer.
        # 考虑由于物质转移导致的donor星自旋角动量变化率
        stars[i].jdot_mt = stars[i].mdot_mt * stars[i].spin * stars[i].R ** 2

        # Spin angular-momentum change of the accretor caused by mass transfer.
        # This depends on whether an accretion disk forms.
        # Estimate r_min to decide whether an accretion disk forms around the accretor.
        # 考虑由于物质转移导致的吸积星自旋角动量变化率, 和吸积盘的是否形成有关
        # 计算rmin以确定在吸积星周围是否会形成吸积盘
        rmin = 0.0425 * self.sep * (q[1 - i] * (1 + q[1 - i])) ** (1 / 4)
        # An accretion disk forms.
        # 存在吸积盘
        if rmin > stars[1 - i].R:
            # Alter spin of the degenerate secondary by assuming that material falls onto the star
            # from the inner edge of a Keplerian accretion disk and that the system is in a steady state.
            term = 2 * np.pi * np.sqrt(stars[1 - i].mass * stars[1 - i].R * period_to_sep ** 3)
            stars[1 - i].jdot_mt = stars[1 - i].mdot_mt * term
        # No accretion disk forms.
        # 不存在吸积盘
        else:
            # Calculate the angular momentum of the transferred material by
            # using the radius of the disk (see Ulrich & Burger) that would have formed if allowed.
            rdisk = 1.7 * rmin
            term = 2 * np.pi * np.sqrt(stars[1 - i].mass * rdisk * period_to_sep ** 3)
            stars[1 - i].jdot_mt = stars[1 - i].mdot_mt * term

        # Enforce angular-momentum conservation between stellar spin and orbital motion.
        # This correction is independent of whether mass transfer is conservative.
        # 考虑双星的自旋角动量和轨道角动量之间的守恒(和物质转移是否守恒无关)
        self.jdot_mt = self.jdot_mt - stars[i].jdot_mt - stars[1 - i].jdot_mt

    # ------------------------------------------------------------------------------------------------------------------
    #                                  Stellar collision without common-envelope evolution
    #                                               恒星碰撞(不经历CE演化)
    # ------------------------------------------------------------------------------------------------------------------
    # Stellar collisions usually occur during unstable mass transfer between stars without extended cores
    # (MS/HeMS/WD/NS/BH), or when an eccentric binary overfills the Roche lobe near periastron.
    # 恒星碰撞通常发生在无核恒星(MS/HeMS/WD/NS/BH)之间的不稳定物质转移或偏心轨道在近心点处充满洛希瓣
    def collision(self, i):
        # Record the binary properties before merger.
        # 记录并合前双星参数
        self.event = 'merge'
        self.save()
        self.update_step()
        self.update_time(use_min_timestep=True)

        # Store the two stars in a list for compact indexing.
        # 添加恒星列表, 方便调用
        stars = [self.star1, self.star2]

        # ******* Merger between two main-sequence-like stars *******
        # ******* 两个主序之间的并合 *******
        # MS + MS / HeMS + HeMS
        if (stars[i].type <= 1 and stars[1 - i].type <= 1) or (stars[i].type == 7 and stars[1 - i].type == 7):
            # Store the information needed to estimate the rejuvenated age.
            # 记录信息以供新年龄的计算
            age_frac = stars[i].age * stars[i].mass / stars[i].tm + stars[1 - i].age * stars[1 - i].mass / stars[
                1 - i].tm
            # Update the stellar type and mass after merger.
            # 更新并合后的恒星类型/质量
            stars[i].type = 1 if stars[i].type <= 1 else 7
            stars[i].mass += stars[1 - i].mass
            stars[i].mass0 = stars[i].mass
            # Recalculate the stellar properties.
            # 更新恒星参数
            stars[i].StellarCal()
            stars[i].age = 0.1 * stars[i].tm * age_frac / stars[i].mass
            stars[i].StellarProp()

        # MS + HeMS
        elif stars[i].type <= 1 and stars[1 - i].type == 7:
            stars[i].type = 4
            stars[i].mass += stars[1 - i].mass
            stars[i].M_core = stars[1 - i].mass
            self.set_new_star(i)
        
        # HeMS + MS (massive helium MS star + extremely low-mass MS companion, which usually forms when α_CE is large.)
        elif stars[i].type == 7 and stars[1 - i].type <= 1:
            stars[i].type = 4
            stars[i].mass += stars[1 - i].mass
            stars[i].M_core = stars[i].mass
            self.set_new_star(i)

        # ******* Merger between a main-sequence-like star and a compact object *******
        # ******* 主序与致密星之间的并合 *******
        # MS + WD
        elif stars[i].type <= 1 and stars[1 - i].type in {10, 11, 12}:
            if stars[1 - i].type == 10:
                stars[i].type = 3
            else:
                stars[i].type = 6
            stars[i].mass += stars[1 - i].mass
            stars[i].M_core = stars[1 - i].mass
            self.set_new_star(i)
        # HeMS + WD
        elif stars[i].type == 7 and stars[1 - i].type in {10, 11, 12}:
            if stars[1 - i].type == 10:
                age_frac = stars[i].age * stars[i].mass / stars[i].tm
                stars[i].type = 7
                stars[i].mass += stars[1 - i].mass
                stars[i].mass0 = stars[i].mass
                stars[i].StellarCal()
                stars[i].age = stars[i].tm * age_frac / stars[i].mass
                stars[i].StellarProp()
            else:
                stars[i].type = 9
                stars[i].mass += stars[1 - i].mass
                stars[i].M_core = stars[1 - i].mass
                self.set_new_star(i)
        # MS/HeMS + NS/BH
        elif stars[i].type in {0, 1, 7} and stars[1 - i].type in {13, 14}:
            # The merger product is treated as an unstable Thorne-Zytkow-like object.
            # It is assumed to eventually leave only the original NS/BH, without significant compact-object mass growth.
            # 并合结果是一个不稳定的Thorne-Zytkow object, 最终只剩下中子星/黑洞, 假设致密星不会增加质量
            stars[i].type = stars[1 - i].type
            stars[i].mass = stars[1 - i].mass

        # ******* Merger between two degenerate objects *******
        # ******* 简并星之间的合并 *******
        # NS/BH + NS/BH
        elif stars[i].type in {13, 14} and stars[1 - i].type in {13, 14}:
            # The merger remnant of a double-NS system may become a BH depending on the total mass.
            # Such systems are associated with short gamma-ray bursts and gravitational-wave events.
            # 双中子星系统并合后根据质量可能会成为黑洞, 对应的观测有短暴、引力波
            mass_total = stars[i].mass + stars[1 - i].mass
            stars[i].type = 13 if mass_total < M_ns_max else 14
            stars[i].mass = mass_total
            stars[i].age = 0
        # WD + NS/BH
        elif stars[i].type in {10, 11, 12} and stars[1 - i].type in {13, 14}:
            # After tidal disruption of a WD by an NS/BH, the material may form an accretion disk
            # or be added to the compact-object mass. This channel may be related to long gamma-ray bursts.
            # 白矮星潮汐瓦解之后, 如果存在吸积盘, 应该会进入吸积盘, 否则应该加到中子星黑洞质量上？有可能发生长伽马暴？
            mass_total = stars[i].mass + stars[1 - i].mass
            stars[i].type = 13 if mass_total < M_ns_max else 14
            stars[i].mass = mass_total
            stars[i].age = 0
        # WD + WD
        elif stars[i].type in {10, 11, 12} and stars[1 - i].type in {10, 11, 12}:
            if stars[i].type == 10 and stars[1 - i].type == 10:
                # HeWD + HeWD
                # Assume the energy released by ignition of the triple-alpha reaction
                # is enough to destroy the star.
                stars[i].type = 15
                stars[i].mass = 0
            elif stars[i].type == 10 and stars[1 - i].type in {11, 12}:
                # HeWD + COWD/ONeWD
                # Should be helium overflowing onto a CO or ONe core in which case the
                # helium swells up to form a giant envelope so a HeGB star is formed.
                # Allowance for the rare case of CO or ONe flowing onto He is made.
                stars[i].type = 9
                stars[i].mass += stars[1 - i].mass
                stars[i].M_core = stars[1 - i].mass
                self.set_new_star(i)
            elif stars[i].type == 11 and stars[1 - i].type == 11:
                # COWD + COWD
                # Simply add the two WD masses, unless the remnant exceeds the Chandrasekhar mass and triggers a Type Ia SN.
                # 简单的质量相加, 除非超过钱德拉塞卡质量发生Ia SN
                stars[i].type = 11
                stars[i].mass += stars[1 - i].mass
                stars[i].age = 0
            else:
                # COWD/ONeWD + ONeWD
                # Simply add the WD masses, unless the remnant exceeds the threshold for AIC.
                # 简单的质量相加, 除非超过钱德拉塞卡质量进行AIC
                stars[i].type = 12
                stars[i].mass += stars[1 - i].mass
                stars[i].age = 0
        # No other collision type should occur.
        # 不应该存在其他的并合
        else:
            raise ValueError(f'The collision type {stars[i].type} and {stars[1 - i].type} is not supported.')

        # Update the stellar properties after merger.
        # 更新并合后的恒星参数
        stars[i].StellarCal()
        stars[i].StellarProp()

        # Check whether the newly formed star immediately undergoes a supernova.
        # 检查新恒星是否发生超新星爆炸
        if stars[i].event in {'AIC', 'ECSN', 'CCSN'}:
            stars[i].SN_kick()

        stars[1 - i].type = 15
        stars[1 - i].mass = 0
        self.event = 'None'
        self.process_disrupted_system()

    # ------------------------------------------------------------------------------------------------------------------
    #                                               Common-envelope evolution
    #                                                     公共包层演化
    # ------------------------------------------------------------------------------------------------------------------
    def CE_evolution(self, i):
        # Record the binary properties before CE evolution.
        # 记录CE前双星参数
        self.event = 'CE'
        self.save()
        self.update_step()
        self.update_time(use_min_timestep=True)

        # Store the two stars in a list for compact indexing.
        # 添加恒星列表, 方便调用
        stars = [self.star1, self.star2]

        # If the donor is a main-sequence star and the companion is giant-like, swap the donor index.
        # This covers rare cases such as mass transfer from a massive HeMS star to a low-mass HeHG companion.
        # 若donor星为主序星, 伴星是类巨星, 需要转换下对象 (极少数情况主序向巨星转移物质, 比如大质量HeMS → 小质量HeHG)
        if stars[i].type in {0, 1, 7}:
            i = 1 - i

        # If a WD fills its Roche lobe outside stable RLOF, it may be treated as a direct merger.
        # 如果白矮星充满洛希瓣且非RLOF, 也直接并合
        # if stars[i].type in {10, 11, 12}:
        #     self.merge(i)
        #     return
        # print(stars[i].type, stars[i].mass, stars[1-i].type, stars[1-i].mass)

        # Calculate the envelope binding energy of the donor.
        # The donor is usually a giant, although rare MS-to-giant mass-transfer cases are also possible.
        # 计算donor的包层结合能(通常为巨星, 极少数为主序星向巨星转移物质)
        stars[i].cal_lambda()
        ebindi = stars[i].mass * (stars[i].mass - stars[i].M_core) / (stars[i].lambda_bind * stars[i].R_mt)

        # If the companion is also a giant, include its envelope binding energy as well.
        # 如果伴星也是巨星, 加上伴星的包层结合能
        if 2 <= stars[1 - i].type <= 9 and stars[1 - i].type != 7:
            stars[1 - i].cal_lambda()
            ebindi += stars[1 - i].mass * (stars[1 - i].mass - stars[1 - i].M_core) / (
                    stars[1 - i].lambda_bind * stars[1 - i].R_mt)

        # Initial orbital energy.
        # 计算初始轨道能
        eorbi = stars[i].mass * stars[1 - i].mass / (2 * self.sep)

        # Eccentric-orbit correction inherited from the BSE treatment.
        # The physical role of this step should be checked further.
        # 考虑偏心轨道【我不明白这一步及后续的意义】
        ecirc = eorbi / (1 - self.ecc ** 2)

        # Final orbital energy if the envelope is successfully ejected.
        # 计算没有合并的最终轨道能量
        eorbf = eorbi + ebindi / alpha_CE

        # Donor mass and radius after CE envelope removal.
        # CE后的主星质量/半径
        stars[i].mass = stars[i].M_core
        stars[i].R = stars[i].R_mt = stars[i].R_core  # 这里修改stars[i].R_mt的原因是为了下一步获得正常的R/R_rl

        # Post-CE orbital separation and companion properties.
        # Companion is a main-sequence-like star or a compact object.
        # CE后的轨道间距和伴星质量/半径
        # 如果伴星是主序星/致密星
        if stars[1 - i].type in {0, 1, 7} or stars[1 - i].type >= 10:
            self.sep = stars[i].M_core * stars[1 - i].mass / (2 * eorbf)
        # Companion is also a giant; remove its envelope as well.
        # 如果伴星是巨星
        else:
            self.sep = stars[i].M_core * stars[1 - i].M_core / (2 * eorbf)
            stars[1 - i].mass = stars[1 - i].M_core
            stars[1 - i].R = stars[1 - i].R_mt = stars[1 - i].R_core

        # Recalculate the Roche-lobe radii and check whether either stellar core still overfills its Roche lobe.
        # 计算双星的洛希瓣半径, 同时检查恒星核是否有充满洛希瓣的情况
        self.cal_radius_rochelobe()

        # Decide whether an HG donor is allowed to survive CE evolution.
        # 是否允许 HG donor 离开CE
        if stars[i].type == 2 and not HG_survive_CE:
            self.CE_merge(i, ebindi, eorbi)
            return

        # CE merge
        if stars[i].R / stars[i].R_rl >= 1 or stars[1 - i].R / stars[1 - i].R_rl >= 1:
            self.CE_merge(i, ebindi, eorbi)
        # CE survive
        else:
            # Eccentricity correction after CE survival, inherited from the BSE treatment.
            # This prescription should be revisited if the CE treatment is updated.
            # 如前所述, 我不明白这里的意义
            if eorbf < ecirc:
                self.ecc = np.sqrt(1 - eorbf / ecirc)
            else:
                self.ecc = 0

            # Update orbital period, angular frequency, and orbital angular momentum.
            # 更新轨道周期/角频率/角动量
            self.totalmass = stars[i].mass + stars[1 - i].mass
            self.period = sep_to_period * (self.sep ** 3 / self.totalmass) ** 0.5
            self.omega = 2 * np.pi / self.period
            reduced_mass = self.star1.mass * self.star2.mass / self.totalmass
            self.jorb = reduced_mass * self.omega * self.sep ** 2 * np.sqrt(1 - self.ecc ** 2)

            # Update stellar properties after CE evolution.
            # 更新CE演化后的恒星参数
            for star in stars:
                star.StellarCal()
                star.StellarProp()
                # Use the physical radius for subsequent spin calculations.
                # 由于要计算后续自旋, 需设置真实半径
                if star.type <= 9:
                    star.R_mt = star.R
                    star.R = min(star.R, star.R_rl) if star.R_rl > 0 else star.R
                else:
                    star.R_mt = star.R

            # Update stellar spins, assuming spin-orbit synchronization after CE.
            # 更新自旋 (假设CE后自转与公转耦合)
            stars[i].spin = stars[1 - i].spin = self.omega
            stars[i].cal_jspin()
            stars[1 - i].cal_jspin()

            # If a supernova occurs after CE, apply the natal kick to the binary orbit.
            # 如果发生超新星爆炸, 考虑natal kick对轨道的影响
            if self.star1.event in {'AIC', 'ECSN', 'CCSN'}:
                self.star1.SN_kick()
            if self.star2.event in {'AIC', 'ECSN', 'CCSN'}:
                self.star2.SN_kick()
            self.check_SN()

    # ------------------------------------------------------------------------------------------------------------------
    #                                           Binary merger during CE evolution
    #                                                   双星在CE中并合
    # ------------------------------------------------------------------------------------------------------------------
    def CE_merge(self, i, ebindi, eorbi):
        # Store the two stars and mass ratios in lists for compact indexing.
        # 添加恒星/质量比列表, 方便调用
        stars = [self.star1, self.star2]
        q = [self.q1, self.q2]

        # If the companion is an NS/BH, assume it does not accrete any envelope or core material.
        # The merger product is treated as an unstable Thorne-Zytkow-like object.
        # 如果是中子星/黑洞, 不吸收任何物质(包层、巨星核), 演化成不稳定TZO
        if stars[1 - i].type in {13, 14}:
            stars[i].type = stars[1 - i].type
            stars[i].mass = stars[1 - i].mass
            stars[i].age = 0
        else:
            # Estimate the total mass of the merger product from the remaining binding energy
            # and the giant mass-radius relation. A limitation here is that the same binding-energy
            # parameter is effectively assumed for all giant envelopes.
            # 下面计算并合后的总质量, 整体思路是计算剩余的结合能+巨星质量-半径关系, 不过这里的不足之处在于假设所有巨星的结合能参数一样

            # First estimate the binding energy immediately before merger from the orbital-energy change
            # at the point where one of the stellar cores just fills its Roche lobe.
            # 首先通过并合前(当某个核恰好充满洛希瓣时)轨道能的变化量计算并合前一刻的结合能
            sep_f = max(stars[i].R / self.rl_ratio(q[i]), stars[1 - i].R / self.rl_ratio(q[1 - i]))
            ebindf = ebindi + alpha_CE * (eorbi - stars[i].mass * stars[1 - i].mass / (2 * sep_f))
            total_mass_before = self.data[self.step - 1]['m1'] + self.data[self.step - 1]['m2']

            # Determine the core mass of the newly formed star.
            # 确定新核的质量
            if stars[i].type <= 6 and stars[1 - i].type <= 1:
                core_new = stars[i].mass
            elif stars[i].type in {8, 9} and stars[1 - i].type <= 1:
                if i == 0:
                    core_new = self.data[self.step - 1]['mc1']
                else:
                    core_new = self.data[self.step - 1]['mc2']
            elif stars[i].type in {8, 9} and stars[1 - i].type == 7:
                core_new = stars[i].mass
            else:
                core_new = stars[i].mass + stars[1 - i].mass
            # print(total_mass_before, self.data[self.step - 1]['m1'], self.data[self.step - 1]['m2'], core_new)
            # print('CE前的结合能', ebindi)
            # print('并合前的结合能', ebindf)
            # print('CE前的轨道能', eorbi)
            # print('并合前的轨道能', stars[i].mass * stars[1 - i].mass / (2 * sep_f))
            # print('并合前的轨道间距', sep_f)
            # print('并合前的总质量', total_mass_before)
            # print('并合前的核质量', core_new)

            # Solve for the total mass after merger using Newton's method following Eq. 77 of Hurley et al. (2002).
            # 下面用牛顿法求解并合后总质量
            const = ebindf / ebindi * total_mass_before ** (1 + stars[i].zpars[7]) * (total_mass_before - core_new)
            Mf = self.solve_merging_mass(a=stars[i].zpars[7], b=core_new, c=const, initial_guess=core_new)
            # try:
            #     const = ebindf / ebindi * total_mass_before ** (1 + stars[i].zpars[7]) * (total_mass_before - core_new)
            #     Mf = self.solve_merging_mass(a=stars[i].zpars[7], b=core_new, c=const, initial_guess=core_new)
            # except Exception as e:
            #     print(f"Error: {e}")
            #     print(stars[i].zpars[7], core_new, const)
            #     print('偏心率:', self.data[0]['ecc'], '周期:', self.data[0]['period'],
            #           '恒星1质量:', self.data[0]['m1'], '恒星2质量:', self.data[0]['m2'],
            #           '恒星1类型:', self.data[0]['type1'], '恒星2类型:', self.data[0]['type2'])

            # Mass and core mass of the newly formed star.
            # 新恒星的质量/核质量
            stars[i].mass = Mf
            stars[i].M_core = core_new

            # Determine the initial mass and age of the newly formed star.
            # 确定新恒星的初始质量和年龄
            if stars[i].type in {2, 3, 4, 5, 6} and stars[1 - i].type <= 1:
                # A main-sequence star is absorbed into the envelope of the giant.
                # In this case the stellar type, initial mass, and age of the giant remnant are left unchanged.
                # 主序星成为巨星包层的一部分, 新恒星的类型/初始质量/年龄无需更改
                pass
            elif stars[i].type in {8, 9} and stars[1 - i].type == 7:
                # Similarly, a helium main-sequence star is absorbed into the envelope of a helium giant.
                # 和上面一样, 氦主序成为氦巨星包层的一部分
                pass
            elif stars[i].type in {8, 9} and stars[1 - i].type <= 1:
                # Helium giant + H-rich main-sequence star -> TPAGB.
                # For this contact merger, assume no mass is lost.
                # 氦巨星 + 氢主序 → TPAGB, 考虑相接双星并合不损失质量
                stars[i].type = 6
                stars[i].mass = total_mass_before
                self.set_new_star(i)
            elif stars[i].type in {8, 9} and stars[1 - i].type == 10:
                stars[i].type = 7
                stars[i].mass0 = stars[i].mass
                stars[i].StellarCal()
                stars[i].age = stars[i].tm * (stars[i].M_core - stars[1 - i].mass) / stars[i].M_core
                stars[i].StellarProp()
            else:
                # Determine the stellar type of the merger product from the merger matrix.
                # 根据并合矩阵确定新恒星的类型
                stars[i].type = self.ktype[stars[1 - i].type, stars[i].type]
                self.set_new_star(i)

        stars[i].StellarCal()
        stars[i].StellarProp()

        if stars[i].event in {'AIC', 'ECSN', 'CCSN'}:
            stars[i].SN_kick()

        stars[1 - i].type = 15
        stars[1 - i].mass = 0
        self.event = 'merge'
        self.process_disrupted_system()

        # The post-merger stellar spin could be tied to the orbital period just before merger,
        # when one of the cores fills its Roche lobe.
        # 假设并合后的恒星自旋周期 = 并合前一刻(有核充满洛希瓣)的轨道周期
        # period = sep_to_period * (sep_f ** 3 / core_new) ** 0.5
        # omega = 2 * np.pi / period

    # ------------------------------------------------------------------------------------------------------------------
    #                                  Set the age and initial mass of a merger product
    #                                              确定并合后新恒星的年龄和初始质量
    # ------------------------------------------------------------------------------------------------------------------
    def set_new_star(self, i):
        star = [self.star1, self.star2][i]
        # Giant branch: for convenience, place the star at the base of the giant branch
        # and solve for a consistent initial mass.
        # 巨星分支, 为了方便, 将演化时刻定格在BGB, 寻找合适的初始质量
        if star.type == 3:
            star.solve_initial_mass_GB(star.M_core)
            star.StellarCal()
            star.age = star.tscls[1] + 1e-6 * (star.tscls[2] - star.tscls[1])
            if star.type == 4:
                star.age = star.tscls[2]
        # Core-helium-burning phase.
        # 氦核燃烧阶段
        elif star.type == 4:
            # For a newly formed CHeB star, estimate its fractional evolutionary age
            # from the burning progress of the two progenitor stars before merger.
            # 对于新的CHeB恒星, 需要知道并合前两个恒星各自的燃烧程度, 以此确定新恒星的年龄比例
            age_frac = 0
            stars = [self.star1, self.star2]
            type_before = [self.data[self.step - 1]['type1'], self.data[self.step - 1]['type2']]
            mass_before = [self.data[self.step - 1]['m1'], self.data[self.step - 1]['m2']]
            M_core_before = [self.data[self.step - 1]['mc1'], self.data[self.step - 1]['mc2']]
            for j in range(2):
                if type_before[j] in {6, 8, 9, 11, 12}:
                    age_frac += M_core_before[j]
                elif type_before[j] in {4, 5}:
                    age_frac += (stars[j].age - stars[j].tscls[2]) / (stars[j].tscls[13] - stars[j].tscls[2]) * \
                                M_core_before[j]
                elif type_before[j] == 7:
                    age_frac += stars[j].age / stars[j].tm * mass_before[j]
                else:
                    pass
            age_frac /= star.M_core
            if age_frac < 0 or age_frac > 1:
                raise ValueError('Age fraction of CHeB star must be between 0 and 1.')
            star.solve_initial_mass_CHeB(star.M_core, age_frac)
            star.StellarCal()
            if star.type == 3:
                star.age = star.tscls[1] + 1e-6 * (star.tscls[2] - star.tscls[1])
            else:
                star.age = star.tscls[2] + age_frac * star.tscls[3]
        # EAGB star.
        # EAGB恒星
        elif star.type == 5:
            star.solve_initial_mass_EAGB(star.M_core)
            star.StellarCal()
            star.age = star.tscls[2] + star.tscls[3]
        # TPAGB star.
        # TPAGB恒星
        elif star.type == 6:
            star.solve_initial_mass_TPAGB(star.M_core)
            star.StellarCal()
            star.age = star.tscls[13]
        # Helium giant.
        # 氦巨星
        elif star.type == 8 or star.type == 9:
            star.solve_initial_mass_HeGB(star.M_core)
            star.StellarCal()
            star.age = star.tscls[1]
        else:
            raise ValueError(
                "Unsupported merged-star type. Expected one of: 3, 4, 5, 6, 8, 9."
            )

    # ------------------------------------------------------------------------------------------------------------------
    #                                             Check for supernova events
    #                                                 检查是否发生超新星爆炸
    # ------------------------------------------------------------------------------------------------------------------
    def check_SN(self):
        # Star 1 undergoes a supernova.
        # 恒星1发生超新星爆炸
        if self.star1.event in {'AIC', 'ECSN', 'CCSN'}:
            self.supernova_in_binary(self.star1, self.star2)
        # Star 2 undergoes a supernova.
        # 恒星2发生超新星爆炸
        elif self.star2.event in {'AIC', 'ECSN', 'CCSN'}:
            self.supernova_in_binary(self.star2, self.star1)

    # ------------------------------------------------------------------------------------------------------------------
    #                                      Effect of supernovae on the binary orbit
    #                                                 超新星对双星轨道的影响
    # ------------------------------------------------------------------------------------------------------------------
    def supernova_in_binary(self, star1, star2):
        star1_mass_old = star1.data[self.step - 1]['mass']

        result = post_supernova_orbit(
            a=self.sep,
            ecc=self.ecc,
            m1_pre=star1_mass_old,
            m2_pre=star2.mass,
            m1_post=star1.mass,
            m2_post=star2.mass,
            kick=star1.v_kick
        )

        state, a, ecc, h_orbit, closest_approach, vc_offset, v1_runaway, v2_runaway, radial_motion = result

        # Binary still bound (any post-SN periastron overflow will be handled by check_overfill() in next step)
        if state == 'bound':
            self.jorb = h_orbit
            self.sep = a
            self.ecc = ecc
            # Record the velocity of the new post-SN binary in the pre-SN center-of-mass frame.
            # The +x direction is defined from the companion toward the exploding star.
            # 记录爆炸后新系统在爆炸前质心静止系内的速度矢量 (伴星指向爆炸星为+x轴)
            self.v_offset = vc_offset
        # Binary is disrupted.
        # 双星瓦解
        elif state == 'disrupted':
            self.event = 'disrupt'
            # Record the runaway velocities after binary disruption.
            # 记录双星瓦解后的逃逸速度
            if self.star1.event in {'AIC', 'ECSN', 'CCSN'}:
                self.v1_offset = v1_runaway
                self.v2_offset = v2_runaway
            else:
                self.v2_offset = v1_runaway
                self.v1_offset = v2_runaway
            self.process_disrupted_system()
        # The orbit disappears, e.g. after a Type Ia SN.
        # 轨道消失(Ia SN)
        else:
            # Record the runaway velocity of the surviving companion.
            # 记录伴星逃逸速度
            if self.star1.event == 'Ia':
                self.v2_offset = v2_runaway
            else:
                self.v1_offset = v2_runaway

            # Update the stellar properties after the SN event.
            # 更新下相关性质
            stars = [self.star1, self.star2]
            for star in stars:
                star.StellarCal()
                star.StellarProp()

    # ------------------------------------------------------------------------------------------------------------------
    #                                      Initialize the merger-product type matrix
    #                                                  初始化ktype矩阵
    # ------------------------------------------------------------------------------------------------------------------
    def _set_ktype(self):
        self.ktype = np.array([[1, 1, 2, 3, 4, 5, 6, 4, 6, 6, 3, 6, 6],
                               [1, 1, 2, 3, 4, 5, 6, 4, 6, 6, 3, 6, 6],
                               [2, 2, 3, 3, 4, 4, 5, 4, 4, 4, 3, 5, 5],
                               [3, 3, 3, 3, 4, 4, 5, 4, 4, 4, 3, 5, 5],
                               [4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4],
                               [5, 5, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4],
                               [6, 6, 5, 5, 4, 4, 6, 4, 6, 6, 5, 6, 6],
                               [4, 4, 4, 4, 4, 4, 4, 7, 8, 9, 7, 9, 9],
                               [6, 6, 4, 4, 4, 4, 6, 8, 8, 9, 7, 9, 9],
                               [6, 6, 4, 4, 4, 4, 6, 9, 9, 9, 7, 9, 9],
                               [3, 3, 3, 3, 4, 4, 5, 7, 7, 7, 15, 9, 9],
                               [6, 6, 5, 5, 4, 4, 6, 9, 9, 9, 9, 11, 12],
                               [6, 6, 5, 5, 4, 4, 6, 9, 9, 9, 9, 12, 12]])

    # ------------------------------------------------------------------------------------------------------------------
    #                                           Initialize orbital parameters
    #                                                  初始化轨道参数
    # ------------------------------------------------------------------------------------------------------------------
    # Set the orbital separation and period. The internal period unit is year.
    # 计算轨道参数 (unit: year)
    def _set_orbital_parameter(self, sep, period):
        if sep > 0:
            self.sep = sep
            self.period = sep_to_period * (self.sep ** 3 / self.totalmass) ** 0.5
        elif period > 0:
            self.period = period / day_per_year
            self.sep = period_to_sep * (self.totalmass * self.period ** 2) ** (1 / 3)
        else:
            raise ValueError("At least one of 'period' and 'separation' must be provided.")

    # Calculate the orbital angular momentum.
    # 计算轨道角动量
    def _set_jorb(self):
        reduced_mass = self.star1.mass * self.star2.mass / self.totalmass
        return reduced_mass * self.omega * self.sep ** 2 * np.sqrt(1 - self.ecc ** 2)

    # ------------------------------------------------------------------------------------------------------------------
    #                                   Set initial stellar spins and spin angular momenta
    #                                          为双星设置合适的初始自旋及角动量
    # ------------------------------------------------------------------------------------------------------------------
    def _set_spin(self):
        if ini_spin_scheme == 'fitting':
            pass
        elif ini_spin_scheme == 'spin-orbit-resonance':
            self.star1.spin = self.omega
            self.star2.spin = self.omega
            self.star1.cal_jspin()
            self.star2.cal_jspin()
        else:
            raise ValueError(
                "Unsupported ini_spin_scheme. Expected one of: 'fitting', 'spin-orbit-resonance'."
            )

    # ------------------------------------------------------------------------------------------------------------------
    #                                            Calculate Roche-lobe radii
    #                                                   计算洛希瓣半径
    # ------------------------------------------------------------------------------------------------------------------
    def cal_radius_rochelobe(self):
        self.q1 = self.star1.mass / self.star2.mass
        self.q2 = self.star2.mass / self.star1.mass
        self.star1.R_rl = self.sep * self.rl_ratio(self.q1)
        self.star2.R_rl = self.sep * self.rl_ratio(self.q2)

    # ------------------------------------------------------------------------------------------------------------------
    #                            Effects of stellar winds on spin, orbit, and eccentricity
    #                                       考虑星风的影响(自旋角动量/轨道角动量/偏心率)
    # ------------------------------------------------------------------------------------------------------------------
    def stellar_wind(self):
        # Compute wind mass-loss rates, wind accretion rates, and total mass-change rates.
        # 星风质量损失率、吸积率、总的变化率
        self.mdot_wind()
        # Spin angular-momentum loss/gain caused by stellar winds.
        # 自旋角动量的变化率
        for i in range(2):
            star1 = [self.star1, self.star2][i]
            star2 = [self.star1, self.star2][1 - i]
            term1 = star1.mdot_wind_loss * star1.spin * star1.R ** 2
            term2 = star1.mdot_wind_acc * star2.spin * star2.R ** 2 * mu_wind
            star1.jdot_wind = (term1 + term2) * (2 / 3)
        # Orbital angular-momentum loss caused by stellar winds.
        # 轨道角动量的变化率
        ecc4 = np.sqrt(1 - self.ecc ** 2)
        term5 = (self.star1.mdot_wind_loss - self.star1.mdot_wind_acc * self.q1) * self.star2.mass ** 2
        term6 = (self.star2.mdot_wind_loss - self.star2.mdot_wind_acc * self.q2) * self.star1.mass ** 2
        self.jdot_wind = (term5 + term6) * self.sep ** 2 * ecc4 * self.omega / self.totalmass ** 2
        # Eccentricity change caused by wind accretion.
        # 偏心率的变化率
        term7 = self.star1.mdot_wind_acc * (0.5 / self.star1.mass + 1.0 / self.totalmass)
        term8 = self.star2.mdot_wind_acc * (0.5 / self.star2.mass + 1.0 / self.totalmass)
        self.edot_wind = -self.ecc * (term7 + term8)

    # Wind mass loss and wind accretion.
    # 星风质量损失/星风吸积
    def mdot_wind(self):
        for i in range(2):
            star1 = [self.star1, self.star2][i]
            star2 = [self.star1, self.star2][1 - i]
            # Compute the wind mass-loss rate of star1 and store it as mdot_wind_loss.
            # 计算 star1 的星风质量损失率，用 mdot_wind_loss 表示
            star1.cal_mdot_wind(self.ecc)

            # Compute the wind-accretion rate of star2 from the wind of star1
            # and store it as mdot_wind_acc (Boffin & Jorissen 1988, A&A, 205, 155).
            # In the expression v = sqrt(GM/R), the simplified code units reduce this to v^2 = M/R.
            # The unit-conversion factors cancel in Eq. 6 of Hurley et al. (2002).
            # 计算 star2 从 star1 星风中质量吸积率, 用 mdot_wind_acc 表示(Boffin & Jorissen, A&A 1988, 205, 155).
            # 在公式v=GM/R中, 可以简化为v=M/R, 单位换算系数在eq.(6) of Hurley et al. 2002中会抵消掉
            vorb2 = (star1.mass + star2.mass) / self.sep
            vwind2 = 2.0 * beta_wind * star1.mass / star1.R
            term1 = 1.0 / np.sqrt(1.0 - self.ecc ** 2)
            term2 = (star2.mass / vwind2) ** 2
            term3 = 1 / (1.0 + vorb2 / vwind2) ** 1.5
            term4 = alpha_wind * abs(star1.mdot_wind_loss) / (2.0 * self.sep ** 2)
            star2.mdot_wind_acc = term1 * term2 * term3 * term4
            star2.mdot_wind_acc = min(star2.mdot_wind_acc, 0.8 * abs(star1.mdot_wind_loss))

    # ------------------------------------------------------------------------------------------------------------------
    #                             Orbital angular-momentum loss from gravitational radiation
    #                                            密近双星的引力波辐射导致轨道角动量损失
    # ------------------------------------------------------------------------------------------------------------------
    def GW_radiation(self):
        ecc4 = np.sqrt(1 - self.ecc ** 2)
        term1 = self.star1.mass * self.star2.mass * self.totalmass / self.sep ** 4
        term2 = (1 + 0.875 * self.ecc ** 2) / ecc4 ** 5
        term3 = ((19 / 6) + (121 / 96) * self.ecc ** 2) / ecc4 ** 5
        self.jdot_gr = - 8.315e-10 * term1 * term2 * self.jorb
        self.edot_gr = - 8.315e-10 * term1 * term3 * self.ecc

    # ------------------------------------------------------------------------------------------------------------------
    #                                                    Tidal effects
    #                                                       潮汐影响
    # ------------------------------------------------------------------------------------------------------------------
    def tide_effect(self, adjustment=False):
        # Reset the tidal terms first, since they need to be recomputed.
        # 由于需要重新调整, 先重置相关变量
        self.jdot_tide = 0
        self.edot_tide = 0

        # For stars close to Roche-lobe filling, include tidal circularization, orbital shrinkage, and spin evolution.
        # 对于非简并星或充满洛希瓣的简并星, 考虑潮汐带来的圆化、轨道收缩和自旋
        for i in range(2):
            star1 = [self.star1, self.star2][i]
            star2 = [self.star1, self.star2][1 - i]

            q = star2.mass / star1.mass
            if (star1.type <= 9 and star1.R >= 0.01 * star1.R_rl) or (star1.type >= 10 and star1.R >= star1.R_rl):
                # In detailed MESA calculations the donor radius only slightly exceeds the Roche-lobe radius,
                # whereas here the stellar radius is not self-consistently adjusted after mass loss.
                # If we used the raw fitted radius directly, it would be spuriously inflated and would transfer
                # too much orbital angular momentum into stellar spin. For this reason, use the Roche-lobe radius
                # as the effective stellar radius in the tidal and wind calculations.
                # 需要注意的是, MESA细致模拟下donor半径只微微超过洛希瓣半径, 而这里的恒星未考虑损失质量后的半径变化,
                # 即恒星半径只依赖单星拟合数据, 这将导致恒星的半径急剧增加, 大量的轨道角动量转移到恒星的自旋角动量,
                # 因此我们可以用洛希瓣半径作为恒星的真实半径代入潮汐及星风计算中
                raa2 = (star1.R / self.sep) ** 2
                raa6 = raa2 ** 3
                ecc2 = self.ecc ** 2
                omecc2 = 1 - self.ecc ** 2
                sqome2 = np.sqrt(1 - self.ecc ** 2)
                sqome3 = sqome2 ** 3
                # Hut's eccentricity polynomials.
                # 赫维茨多项式
                f5 = 1 + ecc2 * (3 + ecc2 * 0.375)
                f4 = 1 + ecc2 * (1.5 + ecc2 * 0.125)
                f3 = 1 + ecc2 * (3.75 + ecc2 * (1.875 + ecc2 * 7.8125e-2))
                f2 = 1 + ecc2 * (7.5 + ecc2 * (5.625 + ecc2 * 0.3125))
                f1 = 1 + ecc2 * (15.5 + ecc2 * (31.875 + ecc2 * (11.5625 + ecc2 * 0.390625)))
                if (star1.type == 1 and star1.mass >= 1.25) or star1.type == 4 or star1.type == 7:
                    # Radiative damping (Zahn, 1977, A&A, 57, 383 and 1975, A&A, 41, 329).
                    # 辐射阻尼
                    tc = 1.592e-9 * (star1.mass ** 2.84)
                    f = 1.9782e4 * np.sqrt((star1.mass * star1.R ** 2) / self.sep ** 5) * tc * (1 + q) ** (5 / 6)
                    tcqr = f * q * raa6
                    rg2 = star1.k2
                elif star1.type <= 9:
                    # Convective damping (Hut, 1981, A&A, 99, 126).
                    # 对流阻尼
                    # Use the effective convective-envelope radius in the tidal calculation.
                    # 如上, 考虑真实的对流包层半径
                    R_conv_env_true = max(1e-10, min(star1.R_conv_env, star1.R - star1.R_core))
                    tc = 0.4311 * (star1.M_conv_env * R_conv_env_true * (star1.R - 0.5 * R_conv_env_true) / (
                            3 * star1.L)) ** (1 / 3)
                    ttid = 2 * np.pi / (1e-10 + abs(self.omega - star1.spin))
                    f = min(1, (ttid / (2 * tc) ** 2))
                    tcqr = 2 * f * q * raa6 * star1.M_conv_env / (21 * tc * star1.mass)
                    rg2 = (star1.k2 * (star1.mass - star1.M_core)) / star1.mass
                    # if i == 0:
                    #     print('tc', star1.M_conv_env, star1.R_conv_env, R_true, star1.L, tc, te, te ** 0.3)
                    #     print('tcqr:', f, q, raa6, star1.M_conv_env, tc, star1.mass)
                else:
                    # Degenerate damping (Campbell, 1984, MNRAS, 207, 433).
                    # 简并阻尼
                    f = 7.33e-9 * (star1.L / star1.mass) ** (5 / 7)
                    tcqr = f * q ** 2 * raa2 ** 2 / (1 + q)
                    rg2 = star1.k3
                # Tidal circularization.
                # 计算圆化
                self.edot_tide += -27 * tcqr * (1 + q) * raa2 * (self.ecc / sqome2 ** 13) * (
                        f3 - (11 / 18) * sqome3 * f4 * star1.spin / self.omega)
                # if self.step > 400:
                #     print('tide:', self.step, self.edot_tide, star1.spin, self.omega, f3 - (11 / 18) * sqome3 * f4 * star1.spin / self.omega)
                tcirc = self.ecc / (abs(self.edot_tide) + 1e-20)
                # Equilibrium spin in the absence of angular-momentum exchange.
                # 计算没有角动量能被转移时的平衡自旋
                spin_eq = self.omega * f2 / (sqome3 * f5)
                # Tidal spin-up/down rate.
                # 计算潮汐引起的自旋变化率
                star1_spin_dot = (3 * q * tcqr / (rg2 * omecc2 ** 6)) * (f2 * self.omega - sqome3 * f5 * star1.spin)
                # Re-limit the synchronization term after the timestep has already been chosen,
                # so that spin-orbit exchange does not overshoot.
                # 重新调整潮汐的同步作用, 防止自旋/轨道角动量之间的过度转移(在确定演化步长后进入此分支)
                if adjustment:
                    if star1_spin_dot >= 0:
                        star1_spin_dot = min(star1_spin_dot, (spin_eq - star1.spin) / self.dt)
                    else:
                        star1_spin_dot = max(star1_spin_dot, (spin_eq - star1.spin) / self.dt)
                # Tidal spin-angular-momentum change.
                # 计算潮汐引起的自旋角动量变化率
                star1.jdot_tide = (star1.k2 * (star1.mass - star1.M_core) * star1.R ** 2 +
                                   star1.k3 * star1.M_core * star1.R_core ** 2) * star1_spin_dot
                # Orbital angular-momentum change through spin-orbit exchange.
                # 计算潮汐造成的轨道角动量变化(与恒星自旋角动量相互转化)
                self.jdot_tide -= star1.jdot_tide

                # print('恒星质量', star1.mass)
                # print(star1.jdot_tide, self.jdot_tide)
                # print(star1_spin_dot, q, tcqr, rg2, self.omega, star1.spin, self.omega - star1.spin)

                # if star1.type <= 6 or abs(djt) / jspin[k] > 0.1:
                #     djtt = djtt + djt

    # ------------------------------------------------------------------------------------------------------------------
    #                                                     Event mapping
    #                                                        事件映射
    # ------------------------------------------------------------------------------------------------------------------
    def event_map(self):
        event_mapping = {
            'CE': b'CE',
            'RLOF begin': b'RLOF begin',
            'RLOF end': b'RLOF end',
            'merge': b'merge',
            'disrupt': b'disrupt',
            'None': b'None'
        }
        return event_mapping.get(self.event)

    # ------------------------------------------------------------------------------------------------------------------
    #                                                   State mapping
    #                                                      状态映射
    # ------------------------------------------------------------------------------------------------------------------
    def state_map(self):
        state_mapping = {
            'detached': b'detached',
            'semidetached': b'semidetached',
            'contact': b'contact',
            'disrupted': b'disrupted'
        }
        return state_mapping.get(self.state)

    @staticmethod
    def rl_ratio(q):
        """Calculate the Roche lobe radius ratio (R_rl / a)

        Args:
            q: Mass ratio (M_secondary / M_primary)

        Returns:
            Roche lobe radius normalized by orbital separation (R_rl / a)

        Notes:
            Eggleton (1983) approximation formula:
            R_rl / a = 0.49 * q^(2/3) / [0.6 * q^(2/3) + ln(1 + q^(1/3))]
        """
        p = q ** (1 / 3)
        rl_div_a = 0.49 * p * p / (0.6 * p * p + np.log(1 + p))
        return rl_div_a

    @staticmethod
    def solve_merging_mass(a, b, c, initial_guess, max_iterations=100000, tolerance=1e-3):
        """Newton's method for solving the total mass after binary merger"""
        def merging_mass(x):
            return x ** (1 + a) * (x - b) - c

        x = initial_guess
        for _ in range(max_iterations):
            f = merging_mass(x)
            if abs(f) < tolerance:
                return x
            df_dx = (2 + a) * x ** (1 + a) - b * (1 + a) * x ** a
            x = x - f / df_dx
        raise ValueError("Merging-mass solver did not converge within max_iterations.")

