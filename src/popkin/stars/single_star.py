import numpy as np
from numba import float64, int64, types, from_dtype
from popkin.metallicity.zcnst import zcnsts_set
from popkin.utils import conditional_jitclass
from popkin.binding_energy import z002, z0001, z00001, lambda_XL2010
from popkin.constants import ahe, aco, tiny
from popkin.constants import period_to_sep, struct_dtype_single
from popkin.constants import sec_per_year, Z_sun, R_sun, T_eff_sun
from popkin.config.controls_default import max_time, max_step, ini_spin_scheme, lambda_binding, alpha_th
from popkin.config.controls_default import (
    SNtype, CCSN_kick_model, sigma_CCSN, CCSN_kick_lognormal_mu, CCSN_kick_lognormal_sigma,
    CCSN_kick_lognormal_vmax, sigma_ECSN, sigma_AIC, M_ECSN, M_ch, M_ns_max, WD_flag, 
    compact_star_max_timestep,
)
from popkin.config.controls_default import mb_model, gamma_mb, wind_model, neta, bwind, f_WR, f_LBV
from popkin.config.user_config import apply_user_config

apply_user_config(globals(), "inlist")


# Single star class
spec = [
    ('type', int64),                        # stellar type
    ('Z', float64),                         # initial mass fraction of metals
    ('mass0', float64),                     # initial mass (solar units)
    ('mass', float64),                      # current mass (solar units)
    ('R', float64),                         # radius (solar units)
    ('L', float64),                         # luminosity (solar units)
    ('dt', float64),                        # 演化时间/evolution timestep       [unit: yr]
    ('Teff', float64),                      # 演化温度/effective temperature [K]
    ('spin', float64),                      # 自旋角频率(unit: /yr)
    ('jspin', float64),                     # 自旋角动量(unit: M_sun * R_sun2 / yr)
    ('M_core', float64),                    # in solar units
    ('M_co_core', float64),                 # in solar units
    ('M_conv_env', float64),                # 对流包层质量(unit: M_sun)
    ('R_core', float64),                    # in solar units
    ('R_conv_env', float64),                # in solar units
    ('R_rl', float64),                      # 洛希瓣半径(unit: R_sun)
    ('R_mt', float64),                      # 用于计算物质转移速率的半径
    ('L_core', float64),                    # in solar units
    ('mdot', float64),                      # 恒星质量变化率
    ('mdot_wind', float64),                 # 恒星星风导致的质量变化率
    ('mdot_wind_loss', float64),            # 星风质量损失率
    ('mdot_wind_acc', float64),             # 星风质量吸积率
    ('mdot_mt', float64),                   # 物质转移速率
    ('jdot', float64),                      # 自旋角动量总变化率变化率
    ('jdot_wind', float64),                 # 星风提取的自旋角动量变化率
    ('jdot_tide', float64),                 # 潮汐引起的自旋角动量变化率
    ('jdot_mt', float64),                   # 物质转移导致的自旋角动量变化率
    ('jdot_mb', float64),                   # 磁制动提取的自旋角动量变化率
    ('time', float64),                      # 当前的演化时间        [unit: yr]
    ('age', float64),                       # 当前type的年龄       [unit: Myr]
    ('step', int64),                        # 当前的演化步数
    ('data', from_dtype(struct_dtype_single)[:]),               # 存储每个步长的属性
    ('zpars', float64[:]),                  # 与金属丰度相关的常数
    ('msp', float64[:]),                    # 主序分支系数
    ('gbp', float64[:]),                    # 巨星分支系数
    ('tm', float64),                        # 主序时间
    ('tn', float64),                        # 核燃烧时间
    ('tscls', float64[:]),                  # 到达不同阶段的时标
    ('lums', float64[:]),                   # 特征光度
    ('GB', float64[:]),                     # 巨星分支参数
    ('f_fb', float64),                      # 超新星爆炸后回落物质所占比例
    ('meanvk', float64),                    # stochastic模型下Natal Kick服从正态分布均值
    ('sigmavk', float64),                   # stochastic模型下Natal Kick数值标准差
    ('rg', float64),                        # giant branch or Hayashi track radius, approporaite for the type.
    ('k3', float64),                        # 恒星核的自旋角动量jspin_core=k3*omega*mc*rc**2
    ('k2', float64),                        # 恒星包层的自旋角动量jspin_envelop=k2*omega*me*re**2
    ('tau_kh', float64),                    # Kelvin-Helmholtz时标
    ('tau_dyn', float64),                   # 动力学时标
    ('lambda_bind', float64),               # 包层结合能lambda
    ('event', types.string),                # 发生的事件
    ('v_kick', float64[:]),                 # 恒星遗迹由于超新星爆炸受到的kick速度
    ('index', int64),                       # 当前双星的编号, 用于确定kick参数
]


@conditional_jitclass(spec)
class SingleStar:
    def __init__(self, type, Z, mass, index=0):
        self.type = type
        self.Z = Z
        self.mass0 = mass
        self.mass = mass
        self.R = 0.
        self.L = 0.
        self.dt = 0.
        self.Teff = T_eff_sun
        self.spin = 0.
        self.jspin = 0.
        self.M_core = 0.
        self.M_co_core = 0.
        self.M_conv_env = 0.
        self.R_core = 0.
        self.R_conv_env = 0.
        self.R_rl = 0.
        self.R_mt = 0.
        self.L_core = 0.
        self.mdot = 0.
        self.mdot_wind = 0.
        self.mdot_wind_loss = 0.
        self.mdot_wind_acc = 0.
        self.mdot_mt = 0.
        self.jdot = 0.
        self.jdot_wind = 0.
        self.jdot_tide = 0.
        self.jdot_mt = 0.
        self.jdot_mb = 0.
        self.time = 0.
        self.age = 0.
        self.step = 0
        self.data = np.zeros(max_step, dtype=struct_dtype_single)
        self.zpars = np.zeros(20)
        self.msp = np.zeros(200)
        self.gbp = np.zeros(200)
        self.tm = 0.
        self.tn = 0.
        self.tscls = np.zeros(20)
        self.lums = np.zeros(10)
        self.GB = np.zeros(20)
        self.f_fb = 0.
        self.meanvk = 0.
        self.sigmavk = 0.
        self.rg = 0.
        self.k3 = 0.21
        self.k2 = 0.21
        self.tau_kh = 1
        self.tau_dyn = 1
        self.lambda_bind = 0.5
        self.event = 'None'
        self.v_kick = np.full(3, np.nan)
        self.index = index
        zcnsts_set(self)           # 设置金属丰度相关常数
        self._set_spin()           # 设置初始自旋
        np.random.seed(index)      # 设置随机数种子, 方便定量分析参数影响

    # ------------------------------------------------------------------------------------------------------------------
    #                                                    演化单星
    # ------------------------------------------------------------------------------------------------------------------
    def evolve(self, loop=True):
        while self.step < max_step:
            # 更新恒星质量/自旋/表面温度/热力学时标/核时标, 以及各种参数
            self.update()

            # 发生了超新星爆炸(单星不可能发生Ia, 双星可能但不会进入此循环, 故不需考虑Ia)
            if self.event in {'AIC', 'ECSN', 'CCSN'}:
                self.dt = 1e-6
                # 正常单星演化
                if loop:
                    self.save()
                    self.reset()
                    self.step += 1
                # 双星瓦解后的单星演化
                else:
                    break

            # 重置变量
            self.reset()

            # 考虑磁制动影响（自旋角动量的减少）
            self.magnetic_braking()

            # 考虑星风的影响（质量/自旋角动量的减少/增加）
            self.stellar_wind()

            # 刷新变量(总的恒星质量/自旋角动量变化率)
            self.refresh()

            # 根据当前阶段确定下一步的步长(yr)
            self.timestep()

            # 对于非致密星, 限制质量损失(<1%)且不超过包层质量
            self.limit_mass_change()

            # 如果是演化双星瓦解后的某个单星, 退出循环, 在双星模块中继续后续演化
            if not loop:
                self.dt = min(self.dt, max_time * 1e6 - self.time, ((self.time // 1e9) + 1) * 1e9 - self.time)
                break

            # 不超过最长演化时间
            if self.time < max_time * 1e6:
                self.dt = min(self.dt, max_time * 1e6 - self.time, ((self.time // 1e9) + 1) * 1e9 - self.time)
            # 如果达到最长演化时间, 结束演化
            else:
                self.finish()
                break

            # 保存恒星的当前属性
            self.save()

            # 更新下一步的演化时间和恒星年龄
            self.time = self.time + self.dt
            self.age = self.age + self.dt / 1e6

            # 更新迭代次数
            self.step = self.step + 1

    # ------------------------------------------------------------------------------------------------------------------
    #                                                     结束演化
    # ------------------------------------------------------------------------------------------------------------------
    def finish(self):
        self.save()
        self.data = self.data[:self.step + 1]

    # ------------------------------------------------------------------------------------------------------------------
    #                                                    保存当前属性
    # ------------------------------------------------------------------------------------------------------------------
    def save(self):
        self.data[self.step]['time'] = self.time / 1e6
        self.data[self.step]['type'] = self.type
        self.data[self.step]['mass'] = self.mass
        self.data[self.step]['M_core'] = self.M_core
        self.data[self.step]['M_conv_env'] = self.M_conv_env
        self.data[self.step]['R'] = self.R_mt
        self.data[self.step]['R_core'] = self.R_core
        self.data[self.step]['R_conv_env'] = self.R_conv_env
        self.data[self.step]['L'] = self.L
        self.data[self.step]['L_core'] = self.L_core
        self.data[self.step]['spin'] = self.spin
        self.data[self.step]['Teff'] = self.Teff
        self.data[self.step]['event'] = self.event_map()
        self.data[self.step]['mdot'] = self.mdot
        self.data[self.step]['mdot_wind'] = self.mdot_wind
        self.data[self.step]['mdot_mt'] = self.mdot_mt
        self.data[self.step]['jspin'] = self.jspin
        self.data[self.step]['jdot'] = self.jdot
        self.data[self.step]['jdot_wind'] = self.jdot_wind
        self.data[self.step]['jdot_tide'] = self.jdot_tide
        self.data[self.step]['jdot_mt'] = self.jdot_mt
        self.data[self.step]['jdot_mb'] = self.jdot_mb
        self.data[self.step]['v_kick_x'] = self.v_kick[0]
        self.data[self.step]['v_kick_y'] = self.v_kick[1]
        self.data[self.step]['v_kick_z'] = self.v_kick[2]

    # ------------------------------------------------------------------------------------------------------------------
    #                                                   更新当前属性
    # ------------------------------------------------------------------------------------------------------------------
    def update(self):
        # 刷新变量(总的质量/自旋角动量变化率)
        self.refresh()

        # 更新质量(如果是主序星, 同时更新初始质量)
        self.mass += self.mdot * self.dt

        # 如果是主序星, 同时更新初始质量和恒星年龄
        if self.type in {0, 1, 7}:
            self.mass0 = self.mass
            # 年龄在当前演化阶段的占比
            age_frac = self.age / self.tm
            self.StellarCal()
            self.age = self.tm * age_frac
            # 对于全对流(<0.35)/有对流核(>1.25)的氢主序星, 增加质量会导致核增长, 从而变得更年轻
            if self.type != 7 and self.mdot > 0 and (self.mass < 0.35 or self.mass > 1.25):
                self.age = self.age * (1 - self.mdot * self.dt / self.mass)
        # 如果是HG恒星, 允许增加初始质量, 但减少初始质量时需检查减少后的新恒星HG末端核质量是否大于当前核质量(小于属于非物理情况, 应当避免),
        # 同时更新恒星年龄
        # 这里我去掉了self.mass0 <= self.zpars[3]的限制, 因为原bse在CE merge中总是更新HG初始质量/年龄, 而且我也找不到限制的理由,
        # 因为大质量和中等质量唯一的区别在于是否出现巨星分支, 并不影响HG的改变
        if self.type == 2:
            mass0_old = self.mass0
            age_frac = (self.age - self.tm) / (self.tscls[1] - self.tm)
            # 假设初始质量更新了, 看在BGB的核质量是否会超过当前核质量, 如果超过, 则放弃更新
            self.mass0 = self.mass
            self.StellarCal()
            # 不可以改变初始质量, 恢复初始质量和恒星属性
            if self.GB[9] < self.M_core:
                self.mass0 = mass0_old
                self.StellarCal()
            # 可以改变初始质量, 同时改变恒星年龄
            else:
                self.age = self.tm + (self.tscls[1] - self.tm) * age_frac

        # 确定恒星的不同演化阶段的时标、标志性光度、巨星分支参数
        self.StellarCal()

        # 确定恒星的光度、半径、核质量、核半径、对流包层质量/半径/转动惯量系数
        self.StellarProp()

        # 如果发生Ia SN, 则退出
        if self.event == 'Ia':
            return

        # 如果发生超新星爆炸, 记录质心速度偏移
        if self.event in {'AIC', 'ECSN', 'CCSN'}:
            self.SN_kick()

        # 对于充满洛希瓣的非简并donor星, 计算半径是拟合半径, 真实半径近似等于洛希瓣半径, 在计算本身性质(自旋、温度、热力学时标)以及
        # 相关演化过程(星风、潮汐)时应采用真实半径, 不过仍应保留拟合半径以计算物质转移速率
        if self.type <= 9:
            self.R_mt = self.R
            self.R = min(self.R, self.R_rl) if self.R_rl > 0 else self.R
        # 对于充满洛希瓣的简并donor星, 真实半径就等于计算半径
        else:
            self.R_mt = self.R

        # 更新自旋角动量/自旋
        self.jspin += self.jdot * self.dt

        # 检查恒星的自旋是否小于零, 如果出现小于零的情况(只会在双星中出现)可能是由于物质/潮汐过度转移
        # 暂时先使用Hurley的方法, 即限定自旋下限, 但自旋/轨道角动量的转移问题没有实际解决
        self.jspin = max(1e-10, self.jspin)

        # 检查恒星自旋是否达到临界转速(对于充满洛希瓣的donor星, 真实半径近似等于洛希瓣半径)
        spin_crit = 2 * np.pi * np.sqrt(self.mass * period_to_sep ** 3 / self.R ** 3)
        jspin_crit = spin_crit * (
                    self.k2 * (self.mass - self.M_core) * self.R ** 2 + self.k3 * self.M_core * self.R_core ** 2)
        self.jspin = min(self.jspin, jspin_crit)
        self.spin = self.jspin / (
                    self.k2 * (self.mass - self.M_core) * self.R ** 2 + self.k3 * self.M_core * self.R_core ** 2)

        # 如果恒星被潮汐加速到临界转速, 那么多余的来自潮汐同步的自旋角动量应回到轨道角动量中
        # if self.jspin > jspin_crit:
        #     # 多余的自旋角动量回到轨道角动量
        #     jspin_excess = min(self.jspin - jspin_crit, self.jdot_tide * self.dt)
        #     self.spin = spin_crit
        #     self.jspin = jspin_crit

        # 更新表面温度
        self.Teff = T_eff_sun * (self.L / self.R ** 2) ** (1 / 4)

        # 更新热力学时标
        self.tau_kh = 3.138e7 * self.mass / (self.R * self.L)
        if self.type in {0, 1, 7} or self.type >= 10:
            self.tau_kh = self.tau_kh * self.mass
        else:
            self.tau_kh = self.tau_kh * (self.mass - self.M_core)

        # 更新动力学时标
        self.tau_dyn = 5.05e-5 * np.sqrt(self.R ** 3 / self.mass)

    # ------------------------------------------------------------------------------------------------------------------
    #                                                      刷新变量
    # ------------------------------------------------------------------------------------------------------------------
    def refresh(self):
        self.mdot_wind = self.mdot_wind_loss + self.mdot_wind_acc
        self.mdot = self.mdot_wind + self.mdot_mt
        self.jdot = self.jdot_mb + self.jdot_wind + self.jdot_mt + self.jdot_tide

    # ------------------------------------------------------------------------------------------------------------------
    #                                                      重置变量
    # ------------------------------------------------------------------------------------------------------------------
    def reset(self):
        self.mdot = self.mdot_wind = self.mdot_wind_loss = self.mdot_wind_acc = self.mdot_mt = 0
        self.jdot = self.jdot_wind = self.jdot_mb = self.jdot_mt = self.jdot_tide = 0
        self.event = 'None'
        self.v_kick = np.full(3, np.nan)

    # ------------------------------------------------------------------------------------------------------------------
    #                                                    事件映射
    # ------------------------------------------------------------------------------------------------------------------
    def event_map(self):
        event_mapping = {
            'AIC': b'AIC',
            'ECSN': b'ECSN',
            'CCSN': b'CCSN',
            'Ia': b'Ia',
            'None': b'None'
        }
        return event_mapping.get(self.event)

    # ------------------------------------------------------------------------------------------------------------------
    #                                            为恒星设置合适的初始自旋及角动量
    # ------------------------------------------------------------------------------------------------------------------
    def _set_spin(self):
        self.StellarCal()
        self.StellarProp()

        # 根据主序星拟合数据设置主序初始自旋(自旋-轨道耦合模型只对双星起作用)
        if ini_spin_scheme in {'fitting', 'spin-orbit-resonance'}:
            vrot = 330 * self.mass0 ** 3.3 / (15 + self.mass0 ** 3.45)
            self.spin = 45.35 * vrot / self.R
        else:
            raise ValueError(
                "Unsupported ini_spin_scheme. Expected one of: 'fitting', 'spin-orbit-resonance'."
            )

        # 计算自旋角动量
        self.cal_jspin()

    # ------------------------------------------------------------------------------------------------------------------
    #                                                     自旋角动量
    # ------------------------------------------------------------------------------------------------------------------
    def cal_jspin(self):
        self.jspin = self.spin * (
                    self.k2 * (self.mass - self.M_core) * self.R ** 2 + self.k3 * self.M_core * self.R_core ** 2)

    # ------------------------------------------------------------------------------------------------------------------
    #                                                      磁制动
    # ------------------------------------------------------------------------------------------------------------------
    # 考虑磁制动的影响
    def magnetic_braking(self):
        # 计算有明显对流包层的恒星因磁制动损失的自旋角动量, 包括主序星(M < 1.25)、靠近巨星分支的HG恒星以及巨星, 不包括完全对流主序星
        if (0.35 < self.mass < 1.25 and self.type <= 1) or 2 <= self.type <= 9:
            if mb_model == 'Rappaport1983':
                self.jdot_mb = -3.8e-30 * self.mass * self.R ** gamma_mb * self.spin ** 3 * R_sun ** 2 / sec_per_year
            elif mb_model == 'Hurley2002':
                self.jdot_mb = -5.83e-16 * self.M_conv_env * (self.R * self.spin) ** 3 / self.mass
            elif mb_model == 'Van2019':
                djmb_Sk = -3.8e-30 * self.mass * self.R ** gamma_mb * self.spin ** 3 * R_sun ** 2 / sec_per_year
                R_conv_env_true = max(1e-10, min(self.R_conv_env, self.R - self.R_core))
                tau_conv = 0.4311 * (
                            self.M_conv_env * R_conv_env_true * (self.R - 0.5 * R_conv_env_true) / (3 * self.L)) ** (
                                       1 / 3)
                tau_conv_sun = 2.8e6 / sec_per_year
                self.jdot_mb = djmb_Sk * (tau_conv / tau_conv_sun) ** 2
            else:
                raise ValueError(
                    "Unsupported mb_model. Expected one of: 'Rappaport1983', 'Hurley2002', 'Van2019'."
                )
        else:
            self.jdot_mb = 0

        # 限制磁制动损失的自旋角动量(<3%), 这可以保证迭代次数不会超过最大值20000, 当然2%也不会影响演化结果
        if self.jdot_mb != 0:
            self.dt = min(self.dt, 0.03 * self.jspin / abs(self.jdot_mb))

    # ------------------------------------------------------------------------------------------------------------------
    #                                               限制质量损失(M_sun/yr)
    # ------------------------------------------------------------------------------------------------------------------
    def limit_mass_change(self):
        self.mdot = self.mdot_wind_loss + self.mdot_wind_acc + self.mdot_mt
        if self.type < 10:
            # 限制 1% 的质量损失
            if abs(self.mdot * self.dt) > 0.01 * self.mass:
                self.dt = min(self.dt, 0.01 * self.mass / abs(self.mdot))

            # 限制每次质量损失不超过包层总质量
            if abs(self.mdot * self.dt) > self.mass - self.M_core:
                self.dt = min(self.dt, (self.mass - self.M_core) / abs(self.mdot))

    # ------------------------------------------------------------------------------------------------------------------
    #                                            考虑星风的影响(质量/自旋角动量)
    # ------------------------------------------------------------------------------------------------------------------
    def stellar_wind(self):
        # 恒星的星风质量损失率, 用 mdot_wind_loss 表示
        self.cal_mdot_wind()

        # 自旋角动量的变化率
        self.jdot_wind = self.mdot_wind_loss * self.spin * self.R ** 2 * (2 / 3)

    # ------------------------------------------------------------------------------------------------------------------
    #                                               总的星风质量损失(M_sun/yr)
    # ------------------------------------------------------------------------------------------------------------------
    # 计算总的星风质量损失
    def cal_mdot_wind(self, ecc=0):
        if wind_model == 'Hurley':
            self.cal_mdot_wind_Hurley(ecc)
        elif wind_model == 'Belczynski':
            self.cal_mdot_wind_Belczynski(ecc)
        else:
            raise ValueError("Unsupported wind_model. Expected one of: 'Hurley', 'Belczynski'.")

    # 计算总的星风质量损失(Hurley模型)
    def cal_mdot_wind_Hurley(self, ecc):
        mdot_NJ = self.cal_mdot_NJ()
        mdot_KR = self.cal_mdot_KR(ecc=ecc)
        mdot_VW = self.cal_mdot_VW()
        mdot_WR = self.cal_mdot_WR()
        mdot_LBV_Hurley = self.cal_mdot_LBV_Hurley()

        if 0 <= self.type <= 6:
            mdot_wind = max(mdot_NJ, mdot_KR, mdot_VW, mdot_WR) + mdot_LBV_Hurley
        elif 7 <= self.type <= 9:
            mdot_wind = max(mdot_NJ, mdot_KR, mdot_WR)
        else:
            mdot_wind = 0
        self.mdot_wind_loss = -mdot_wind

    # 计算总的星风质量损失(Belczynski模型)
    def cal_mdot_wind_Belczynski(self, ecc):
        mdot_OB = self.cal_mdot_OB()
        mdot_KR = self.cal_mdot_KR(ecc=ecc)
        mdot_WR = self.cal_mdot_WR(z_dependent=True)
        mdot_LBV_Belczynski = self.cal_mdot_LBV_Belczynski()

        # LBV星
        if mdot_LBV_Belczynski > 0:
            self.mdot_wind_loss = -mdot_LBV_Belczynski
        # 氦星
        elif 7 <= self.type <= 9:
            self.mdot_wind_loss = -max(mdot_KR, mdot_WR)
        # OB星
        elif mdot_OB > 0:
            self.mdot_wind_loss = -mdot_OB
        # 其他情况
        else:
            self.cal_mdot_wind_Hurley(ecc=ecc)

    # -------------------------------------------------------------------------------------------------------------------
    #                                              各种星风质量损失(M_sun/yr)
    # -------------------------------------------------------------------------------------------------------------------
    # calculate mass loss rate for massive stars (L > 4000L_sun) over the entire HRD
    # Nieuwenhuijzen & de Jager 1990, A&A, 231, 134
    def cal_mdot_NJ(self):
        if self.L > 4000:
            term1 = 9.631e-15 * min(1.0, (self.L - 4000) / 500)
            term2 = self.R ** 0.81 * self.L ** 1.24 * self.mass ** 0.16 * (self.Z / Z_sun) ** 0.5
            mdot_NJ = term1 * term2
        else:
            mdot_NJ = 0
        return mdot_NJ

    # Calculate mass loss rate for massive OB stars using the Vink et al. 2001 prescription
    # Vink et al. 2001, eqs 24 & 25; Belczynski et al. 2010, eqs 6 & 7
    def cal_mdot_OB(self):
        if self.type <= 6 and 1.25e4 < self.Teff <= 2.5e4:
            term1 = - 6.688 + 2.21 * np.log10(self.L / 1.0e5)
            term2 = - 1.339 * np.log10(self.mass / 30) - 1.601 * np.log10(1.3 / 2)
            term3 = 1.07 * np.log10(self.Teff / 2e4) + 0.85 * np.log10(self.Z / Z_sun)
            mdot_OB = 10 ** (term1 + term2 + term3)
        elif self.type <= 6 and 2.5e4 < self.Teff <= 5.0e4:
            term1 = - 6.697 + 2.194 * np.log10(self.L / 1.0e5) - 1.313 * np.log10(self.mass / 30.0)
            term2 = - 1.226 * np.log10(2.6 / 2.0) + 0.933 * np.log10(self.Teff / 4.0e4)
            term3 = - 10.92 * np.log10(self.Teff / 4.0e4) ** 2 + 0.85 * np.log10(self.Z / Z_sun)
            mdot_OB = 10 ** (term1 + term2 + term3)
        else:
            mdot_OB = 0
        return mdot_OB

    # calculate mass loss rate on the GB and beyond
    # Hurley et al. 2000, eq 106 (based on a prescription taken from Kudritzki & Reimers, 1978, A&A, 70, 227)
    def cal_mdot_KR(self, ecc):
        if 2 <= self.type <= 9:
            mdot_KR = neta * 4e-13 * self.R * self.L / self.mass
            # 考虑 mdot_KR 受潮汐增强(如果应用, 这里可能还需要考虑偏心轨道的情况)
            if self.R_rl > 0.0 and 0 <= ecc < 1:
                rochelobe_periastron = self.R_rl * (1.0 - ecc)
                mdot_KR = mdot_KR * (1.0 + bwind * (min(0.5, (self.R / rochelobe_periastron))) ** 6)
        else:
            mdot_KR = 0
        return mdot_KR

    # calculate mass loss rate on the AGB based on the Mira pulsation period
    # Hurley et al. 2000, just after eq 106 (from Vassiliadis & Wood, 1993, ApJ, 413, 641)
    def cal_mdot_VW(self):
        if 5 <= self.type <= 6:
            p0 = min(1995, 8.51e-3 * self.R ** 1.94 / self.mass ** 0.9)
            p1 = 100 * max(self.mass - 2.5, 0)
            mdot_VW = min(10 ** (-11.4 + 0.0125 * (p0 - p1)), 1.36e-9 * self.L)
        else:
            mdot_VW = 0
        return mdot_VW

    # calculate mass loss of Wolf–Rayet or Wolf–Rayet like star with small H-envelope mass
    # Hurley et al. 2000, just after eq 106 (taken from Hamann, Koesterke & Wessolowski 1995, Hamann & Koesterke 1998)
    # Belczynski et al. 2010, eq 9 when z_dependent is True
    def cal_mdot_WR(self, z_dependent=False):
        mdot_WR = f_WR * 1e-13 * self.L ** 1.5
        if 0 <= self.type <= 6:
            lum0 = 7e4
            kap = -0.5
            mu = (self.mass - self.M_core) / self.mass * min(5.0, max(1.2, (self.L / lum0) ** kap))
            mdot_WR = mdot_WR * (1 - mu) if mu < 1.0 else 0
        elif 7 <= self.type <= 9:
            mdot_WR = mdot_WR * (self.Z / Z_sun) ** 0.86 if z_dependent else mdot_WR
        else:
            mdot_WR = 0
        return mdot_WR

    # Calculate LBV-like mass loss rate for stars beyond the Humphreys-Davidson limit (Humphreys & Davidson 1994)
    # Hurley+ 2000 Section 7.1 a few equation after Eq. 106 (Equation not labelled)
    def cal_mdot_LBV_Hurley(self):
        HD = 1e-5 * self.R * self.L ** 0.5
        if self.L > 6e5 and HD > 1:
            mdot_LBV_Hurley = 0.1 * (HD - 1) ** 3 * (self.L / 6e5 - 1)
        else:
            mdot_LBV_Hurley = 0
        return mdot_LBV_Hurley

    # Calculate LBV-like mass loss rate for stars beyond the Humphreys-Davidson limit (Humphreys & Davidson 1994)
    # Belczynski et al. 2010, eq 8
    def cal_mdot_LBV_Belczynski(self):
        HD = 1e-5 * self.R * self.L ** 0.5
        if self.L > 6e5 and HD > 1.0:
            mdot_LBV_Belczynski = f_LBV * 1e-4
        else:
            mdot_LBV_Belczynski = 0
        return mdot_LBV_Belczynski

    # ------------------------------------------------------------------------------------------------------------------
    #                                               确定恒星演化的更新步长
    # ------------------------------------------------------------------------------------------------------------------
    def timestep(self):
        # 控制各个阶段的步长大小
        pts1 = 0.04      # MS       # 从0.05 → 0.04, 仅针对极少数系统(1/100000)做的优化
        pts2 = 0.01      # CHeB, GB, AGB, HeGB
        pts3 = 0.02      # HG, HeMS

        if self.type <= 1:
            dt = pts1 * self.tm
            dtr = self.tm - self.age
        elif self.type == 2:
            dt = pts3 * (self.tscls[1] - self.tm)      # 【更改】把这里的 pts1 改成 pts3, 缩短 HG 的演化步长
            dtr = self.tscls[1] - self.age
        elif self.type == 3:
            if self.age < self.tscls[6]:
                dt = pts2 * (self.tscls[4] - self.age)
            else:
                dt = pts2 * (self.tscls[5] - self.age)
            dtr = np.minimum(self.tscls[2], self.tn) - self.age
        elif self.type == 4:
            dt = pts2 * self.tscls[3]
            dtr = np.minimum(self.tn, self.tscls[2] + self.tscls[3]) - self.age
        elif self.type == 5:
            if self.age < self.tscls[9]:
                dt = pts3 * (self.tscls[7] - self.age)
            else:
                dt = pts3 * (self.tscls[8] - self.age)
            dtr = np.minimum(self.tn, self.tscls[13]) - self.age
        elif self.type == 6:
            if self.age < self.tscls[12]:
                dt = pts3 * (self.tscls[10] - self.age)
            else:
                dt = pts3 * (self.tscls[11] - self.age)
            dt = np.minimum(dt, 0.005)
            dtr = self.tn - self.age
        elif self.type == 7:
            dt = pts1 * self.tm
            dtr = self.tm - self.age
        elif self.type == 8 or self.type == 9:
            if self.age < self.tscls[6]:
                dt = pts2 * (self.tscls[4] - self.age)
            else:
                dt = pts2 * (self.tscls[5] - self.age)
            dtr = self.tn - self.age
        else:
            dt = min(max(0.1, self.dt * 10 / 1e6), compact_star_max_timestep)
            dtr = dt

        self.dt = max(0.1, min(dt, dtr) * 1e6)

    # ------------------------------------------------------------------------------------------------------------------
    #                                 根据各种CCSN超新星模型, 确定爆炸后的致密星类型(NS/BH)和遗迹质量
    # ------------------------------------------------------------------------------------------------------------------
    def SN_remnant(self, mcbagb):
        self.event = 'CCSN'

        # 当前的核质量, 通常为SN爆发前的CO核质量
        # 变量 mcbagb 表示bagb时的氦核质量(包括He+CO核)或氦星的当前质量, 仅用于stochastic模型的计算

        if SNtype == 'rapid':  # rapid SN, origin from Fryer et al. 2012, ApJ, 749, 91
            self.SN_remnant_rapid()
        elif SNtype == 'delayed':  # delayed SN, origin from Fryer et al. 2012, ApJ, 749, 91
            self.SN_remnant_delayed()
        elif SNtype == 'stochastic':  # stochastic SN, origin from Mandel et al. 2020, MNRAS 499, 3214–3221
            self.SN_remnant_stochastic(mcbagb)
        else:
            raise ValueError("Unsupported SNtype. Expected one of: 'rapid', 'delayed', 'stochastic'.")

    def SN_remnant_rapid(self):
        mproto = 1.0
        if self.M_core < 2.5:
            mfb = 0.2
        elif 2.5 <= self.M_core < 6:
            mfb = 0.286 * self.M_core - 0.514
        elif 6 <= self.M_core < 7:
            mfb = self.mass - mproto
        elif 7 <= self.M_core < 11:
            a1 = 0.25 - 1.275 / (self.mass - mproto)
            b1 = -11.0 * a1 + 1.0
            mfb = (self.mass - mproto) * (a1 * self.M_core + b1)
        else:
            mfb = self.mass - mproto
        self.f_fb = mfb / (self.mass - mproto)
        mrem_bar = mfb + mproto  # 遗迹重子质量
        mrem1 = -6.6667 + 0.6667 * (100 + 30 * mrem_bar) ** 0.5  # 中子星引力质量
        mrem2 = 0.9 * mrem_bar  # 黑洞引力质量
        # 中子星
        if mrem1 <= M_ns_max:
            self.type = 13
            self.mass = mrem1
        # 黑洞
        else:
            self.type = 14
            self.mass = mrem2

    def SN_remnant_delayed(self):
        if self.M_core <= 3.5:
            mproto = 1.2
        elif 3.5 < self.M_core <= 6.0:
            mproto = 1.3
        elif 6.0 < self.M_core <= 11.0:
            mproto = 1.4
        else:
            mproto = 1.6
        if self.M_core <= 2.5:
            mfb = 0.2
        elif 2.5 < self.M_core <= 3.5:
            mfb = 0.5 * self.M_core - 1.05
        elif 3.5 < self.M_core <= 11.0:
            a2 = 0.133 - 0.093 / (self.mass - mproto)
            b2 = -11.0 * a2 + 1.0
            mfb = (self.mass - mproto) * (a2 * self.M_core + b2)
        else:
            mfb = self.mass - mproto
        self.f_fb = mfb / (self.mass - mproto)
        mrem_bar = mfb + mproto  # 遗迹重子质量
        mrem1 = -6.6667 + 0.6667 * (100 + 30 * mrem_bar) ** 0.5  # 中子星引力质量
        mrem2 = 0.9 * mrem_bar  # 黑洞引力质量
        # 中子星
        if mrem1 <= M_ns_max:
            self.type = 13
            self.mass = mrem1
        # 黑洞
        else:
            self.type = 14
            self.mass = mrem2

    def SN_remnant_stochastic(self, mcbagb):
        m11 = 2.0
        m22 = 3.0
        m33 = 7.0
        m44 = 8.0
        meanbh = 0.8
        sigmabh = 0.5
        p1 = np.random.random()
        p2 = np.random.random()

        # 计算黑洞形成时物质完全回落(complete fallback)的概率
        if m11 <= self.M_core < m44:
            pcf = (self.M_core - m11) / (m44 - m11)
        else:
            pcf = 1.0
        # 中子星
        if self.M_core < m11:
            mean0 = 1.2
            sigma0 = 0.02
            self.type = 13
            self.mass = np.random.normal(mean0, sigma0)
        # 中子星或黑洞
        elif m11 <= self.M_core < m33:
            # 计算遗迹是黑洞的概率
            pbh = (self.M_core - m11) / (m33 - m11)
            # 黑洞
            if p1 <= pbh:
                self.type = 14
                # 完全回落
                if p2 <= pcf:
                    self.mass = mcbagb
                # 不完全回落
                else:
                    self.mass = np.random.normal(meanbh * self.M_core, sigmabh)
            # 中子星
            else:
                self.type = 13
                if m11 <= self.M_core < m22:
                    mean0 = 1.4 + 0.5 * (self.M_core - m11) / (m22 - m11)
                    sigma0 = 0.05
                else:
                    mean0 = 1.4 + 0.4 * (self.M_core - m22) / (m33 - m22)
                    sigma0 = 0.05
                self.mass = np.random.normal(mean0, sigma0)
        # 黑洞
        else:
            self.type = 14
            # 完全回落
            if p2 <= pcf:
                self.mass = mcbagb
            # 不完全回落
            else:
                self.mass = np.random.normal(meanbh * self.M_core, sigmabh)
        # 对于 stochastic SN, 速度踢服从一定的正态分布(高斯分布)
        if self.type == 13:
            self.mass = min(max(1.13, self.mass), 2)
            self.meanvk = 520.0 * (self.M_core - self.mass) / self.mass
            self.sigmavk = 0.3 * self.meanvk
        elif self.type == 14:
            self.mass = min(max(2.0, self.mass), mcbagb)
            self.meanvk = 200.0 * max((self.M_core - self.mass) / self.mass, 0.)
            self.sigmavk = 0.3 * self.meanvk


    # ------------------------------------------------------------------------------------------------------------------
    #                                              恒星在爆炸后受到的kick速度
    # ------------------------------------------------------------------------------------------------------------------
    def SN_kick(self):
        # 通过AIC形成的中子星/黑洞
        if self.event == 'AIC':
            self.v_kick = sigma_AIC * np.random.standard_normal(size=3)
        # 通过ECSN形成的中子星
        elif self.event == 'ECSN':
            self.v_kick = sigma_ECSN * np.random.standard_normal(size=3)
        # 通过CCSN形成的中子星/黑洞
        elif self.event == 'CCSN':
            # 对于rapid/delayed SN, natal kick由CCSN_kick_model控制
            if SNtype == 'rapid' or SNtype == 'delayed':
                if CCSN_kick_model == 'hobbs2005':
                    # Hobbs et al. 2005: three Gaussian components, Maxwellian speed
                    self.v_kick = sigma_CCSN * np.random.standard_normal(size=3)
                elif CCSN_kick_model == 'disberg2025':
                    # Disberg & Mandel 2025: lognormal speed with isotropic direction
                    v_kick_magnitude = np.random.lognormal(
                        CCSN_kick_lognormal_mu,
                        CCSN_kick_lognormal_sigma,
                    )
                    if CCSN_kick_lognormal_vmax is not None:
                        while v_kick_magnitude > CCSN_kick_lognormal_vmax:
                            v_kick_magnitude = np.random.lognormal(
                                CCSN_kick_lognormal_mu,
                                CCSN_kick_lognormal_sigma,
                            )

                    phi = np.random.uniform(0.0, 2.0 * np.pi)
                    cos_theta = np.random.uniform(-1.0, 1.0)
                    sin_theta = np.sqrt(1.0 - cos_theta * cos_theta)
                    self.v_kick = v_kick_magnitude * np.array([
                        cos_theta,
                        sin_theta * np.cos(phi),
                        sin_theta * np.sin(phi),
                    ])
                else:
                    raise ValueError("Unsupported CCSN_kick_model. Expected one of: 'hobbs2005', 'disberg2025'.")

                # 对于黑洞, 受到的速度踢在中子星的基础上乘上一个回落因子
                if self.type == 14:
                    self.v_kick = self.v_kick * (1 - self.f_fb)
            # 对于 stochastic SN, natal kick服从一定的正态分布(高斯分布)
            elif SNtype == 'stochastic':
                # kick大小
                v_kick_magnitude = np.random.normal(self.meanvk, self.sigmavk)
                # kick方向
                phi = np.random.uniform(0, 2 * np.pi)
                theta = np.arccos(np.random.uniform(-1, 1))  # 计算 theta 角度
                # kick坐标
                x = np.cos(theta)
                y = np.sin(theta) * np.cos(phi)
                z = np.sin(theta) * np.sin(phi)
                # kick矢量
                self.v_kick = v_kick_magnitude * np.array([x, y, z])
            else:
                raise ValueError("Unsupported SNtype. Expected one of: 'rapid', 'delayed', 'stochastic'.")
        else:
            raise ValueError("Unsupported supernova event. Expected one of: 'AIC', 'ECSN', 'CCSN'.")


    # ------------------------------------------------------------------------------------------------------------------
    #                                               确定包层结合能参数lambda
    # ------------------------------------------------------------------------------------------------------------------
    def cal_lambda(self):
        # 对于氦星, 简单假设结合能参数为0.5
        if self.type >= 7:
            self.lambda_bind = 0.5
            return

        # 富氢恒星的结合能参数计算
        if lambda_binding == 'WJL2016':
            self.cal_lambda_WJL2016()
        elif lambda_binding == 'XL2010':
            self.cal_lambda_XL2010()
        else:
            raise ValueError("Unsupported lambda_binding. Expected one of: 'WJL2016', 'XL2010'.")

    def cal_lambda_WJL2016(self):
        arr = np.array([0.06, 1.5, 3, 5, 7, 9, 15, 25, 35, 50])
        idx = np.searchsorted(arr, self.mass0) - 1

        # 当z=0.02时
        r_z002 = z002[idx][:2000, 0]
        lg_z002 = z002[idx][:2000, 1]
        lb_z002 = z002[idx][:2000, 2]
        # lambda_z002 = lb_z002[max(int(0), np.searchsorted(r_z002, self.R_mt) - 1)]
        if self.R_mt <= r_z002[0]:
            lambda_z002 = lb_z002[0]
        elif self.R_mt >= r_z002[1999]:
            lambda_z002 = lb_z002[1999]
        else:
            lambda_z002 = lb_z002[np.where(r_z002 < self.R_mt)[0][-1]]

        # 当z=0.001时
        r_z0001 = z0001[idx][:2000, 0]
        lg_z0001 = z0001[idx][:2000, 1]
        lb_z0001 = z0001[idx][:2000, 2]
        # lambda_z0001 = lb_z0001[max(int(0), np.searchsorted(r_z0001, self.R_mt) - 1)]
        if self.R_mt <= r_z0001[0]:
            lambda_z0001 = lb_z0001[0]
        elif self.R_mt >= r_z0001[1999]:
            lambda_z0001 = lb_z0001[1999]
        else:
            lambda_z0001 = lb_z0001[np.where(r_z0001 < self.R_mt)[0][-1]]

        # 当z=0.0001时
        r_z00001 = z00001[idx][:2000, 0]
        lg_z00001 = z00001[idx][:2000, 1]
        lb_z00001 = z00001[idx][:2000, 2]
        # lambda_z00001 = lb_z00001[max(int(0), np.searchsorted(r_z00001, self.R_mt) - 1)]
        if self.R_mt <= r_z00001[0]:
            lambda_z00001 = lb_z00001[0]
        elif self.R_mt >= r_z00001[1999]:
            lambda_z00001 = lb_z00001[1999]
        else:
            lambda_z00001 = lb_z00001[np.where(r_z00001 < self.R_mt)[0][-1]]

        # 根据金属丰度计算结合能参数
        if self.Z > 0.02:
            self.lambda_bind = lambda_z002
        elif 0.001 < self.Z <= 0.02:
            self.lambda_bind = lambda_z002 + (0.02 - self.Z) / (0.02 - 0.001) * (lambda_z0001 - lambda_z002)
        elif 0.0001 <= self.Z <= 0.001:
            self.lambda_bind = lambda_z0001 + (0.001 - self.Z) / (0.001 - 0.0001) * (lambda_z00001 - lambda_z0001)
        else:
            raise ValueError(
                "Unsupported metallicity for WJL2016 lambda tables. Expected Z >= 0.0001; Z > 0.02 is clamped to 0.02."
            )

    def cal_lambda_XL2010(self):
        arr = np.array([0.06, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5, 11, 13, 15, 18, 35, 75])
        masses = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 14, 16, 20, 50, 100])
        idx = np.searchsorted(arr, self.mass0) - 1
        mass0 = masses[idx]

        # 检查恒星所处阶段
        if self.type in {2, 3}:
            stage = 1
        elif self.type in {4}:
            stage = 2
        elif self.type in {5, 6}:
            stage = 3
        else:
            raise ValueError(
                "Unsupported stellar type for binding-energy lambda calculation. Expected one of: 2, 3, 4, 5, 6."
            )

        # 计算包层质量分数
        m_env = (self.mass - self.M_core) / self.mass

        # 当z=0.02时
        lambda_b, lambda_g = lambda_XL2010(Z=0.02, stage=stage, mass0=mass0, R=self.R_mt, m_env=m_env)
        lambda_002 = lambda_b * alpha_th + lambda_g * (1 - alpha_th)

        # 当z=0.001时
        lambda_b, lambda_g = lambda_XL2010(Z=0.001, stage=stage, mass0=mass0, R=self.R_mt, m_env=m_env)
        lambda_0001 = lambda_b * alpha_th + lambda_g * (1 - alpha_th)

        # 对真实的z进行插值
        if self.Z >= 0.02:
            self.lambda_bind = lambda_002
        elif self.Z <= 0.001:
            self.lambda_bind = lambda_0001
        else:
            w = (self.Z - 0.001) / (0.02 - 0.001)
            self.lambda_bind = lambda_0001 + w * (lambda_002 - lambda_0001)

    # ------------------------------------------------------------------------------------------------------------------
    #                                     推导不同演化阶段的典型时标、特征光度、巨星分支参数
    #
    # 输入参数 type, mass0, mass
    #
    # 输出参数
    #       tm = 0                                 # 主序时间
    #       tn = 0                                 # 核燃烧时间
    #       tscls = np.zeros((1, 21)).flatten()    # 到达不同阶段的时标
    #       lums = np.zeros((1, 11)).flatten()     # 特征光度
    #       GB = np.zeros((1, 11)).flatten()       # 巨星分支参数
    #
    #       [tscls] 1: BGB               2: He ignition         3: He burning      (BGB is the base of giant branch.)
    #               4: Giant t(inf1)     5: Giant t(inf2)       6: Giant t(Mx)
    #               7: EAGB t(inf1)      8: EAGB t(inf2)        9: EAGB  t(Mx)
    #               10: TPAGB t(inf1)    11: TPAGB t(inf2)      12: TPAGB  t(Mx)
    #               13: TP               14: t(Mcmax)                              (TP is thermally-pulsing AGB)
    #
    #       [lums]  1: ZAMS              2: End MS              3: BGB
    #               4: He ignition       5: He burning          6: L(Mx)
    #               7: BAGB              8: TP
    #
    #       [GB]    1: effective A(H)    2: A(H,He)             3: B
    #               4: D                 5: p                   6: q
    #               7: Mx                8: A(He)               9: Mc,BGB
    #
    # ------------------------------------------------------------------------------------------------------------------
    def StellarCal(self):
        # 限制非简并星的初始质量, 防止超过拟合范围
        if self.mass0 > 100 and self.type < 10:
            self.mass0 = 100

        if self.type <= 6:
            self.StellarCal_H_star()
        elif self.type <= 9:
            self.StellarCal_He_star()
        elif self.type <= 15:
            self.StellarCal_CO()
        else:
            raise ValueError("Unsupported stellar type. Expected an integer type in the range 0..15.")

    # ------------------------------------------------------------------------------------------------------------------
    #                                      推导富氢恒星的典型时标、特征光度、巨星分支参数
    # ------------------------------------------------------------------------------------------------------------------
    def StellarCal_H_star(self):
        # 主序和 BGB 时间
        self.tscls[1] = self.tbgbf()
        self.tm = np.maximum(self.zpars[8], self.thook_div_tBGB()) * self.tscls[1]
        # 零龄主序和主序末尾的光度
        self.lums[1] = self.l_zams()
        self.lums[2] = self.ltmsf()
        # 设置巨星分支参数 GB
        self.GB[1] = 10 ** max(-4.8, min(-5.7 + 0.8 * self.mass0, -4.1 + 0.14 * self.mass0))
        self.GB[2] = 1.27e-5
        self.GB[8] = 8e-5
        self.GB[3] = max(3e4, 500 + 1.75e4 * self.mass0 ** 0.6)
        if self.mass0 <= self.zpars[2]:
            self.GB[4] = self.zpars[6]
            self.GB[5] = 6
            self.GB[6] = 3
        elif self.mass0 < 2.5:
            # 这里用的是线性插值，很明显在 mass=2.5 处，self.GB[4] = 0.975 * zcnsts.zpars[6] - 0.18 * mass
            dlogD = (0.975 * self.zpars[6] - 0.18 * 2.5) - self.zpars[6]
            self.GB[4] = self.zpars[6] + dlogD * (self.mass0 - self.zpars[2]) / (2.5 - self.zpars[2])
            self.GB[5] = 6 - (self.mass0 - self.zpars[2]) / (2.5 - self.zpars[2])
            self.GB[6] = 3 - (self.mass0 - self.zpars[2]) / (2.5 - self.zpars[2])
        else:
            self.GB[4] = max(-1, 0.975 * self.zpars[6] - 0.18 * self.mass0, 0.5 * self.zpars[6] - 0.06 * self.mass0)
            self.GB[5] = 5
            self.GB[6] = 2
        self.GB[4] = 10 ** self.GB[4]
        self.GB[7] = (self.GB[3] / self.GB[4]) ** (1 / (self.GB[5] - self.GB[6]))
        # Change in slope of giant L-Mc relation.
        self.lums[6] = self.GB[4] * self.GB[7] ** self.GB[5]
        # 氦点燃光度
        self.lums[4] = self.lHeIf()
        self.lums[7] = self.lbagbf()
        if self.mass0 < 0.1 and self.type <= 1:
            self.tscls[2] = 1.1 * self.tscls[1]
            self.tscls[3] = 0.1 * self.tscls[1]
            self.lums[3] = self.l_bgb()
            self.tn = 1e10
            return

        # 中小质量恒星, 会经历FGB阶段
        if self.mass0 <= self.zpars[3]:
            # 巨星分支底部的光度
            self.lums[3] = self.l_bgb()
            # Set GB timescales
            self.tscls[4] = self.tscls[1] + (1 / ((self.GB[5] - 1) * self.GB[1] * self.GB[4])) * (
                        (self.GB[4] / self.lums[3]) ** ((self.GB[5] - 1) / self.GB[5]))
            self.tscls[6] = self.tscls[4] - (self.tscls[4] - self.tscls[1]) * (
                        (self.lums[3] / self.lums[6]) ** ((self.GB[5] - 1) / self.GB[5]))
            self.tscls[5] = self.tscls[6] + (1 / ((self.GB[6] - 1) * self.GB[1] * self.GB[3])) * (
                        (self.GB[3] / self.lums[6]) ** ((self.GB[6] - 1) / self.GB[6]))
            # 设置氦点燃时间
            if self.lums[4] <= self.lums[6]:
                self.tscls[2] = self.tscls[4] - (1 / ((self.GB[5] - 1) * self.GB[1] * self.GB[4])) * (
                            (self.GB[4] / self.lums[4]) ** ((self.GB[5] - 1) / self.GB[5]))
            else:
                self.tscls[2] = self.tscls[5] - (1 / ((self.GB[6] - 1) * self.GB[1] * self.GB[3])) * (
                            (self.GB[3] / self.lums[4]) ** ((self.GB[6] - 1) / self.GB[6]))
            # 小质量恒星
            if self.mass0 <= self.zpars[2]:
                mc1 = self.lum_to_mc_gb(self.lums[4])
                self.lums[5] = self.lzahbf(self.mass0, mc1, self.zpars[2])
                self.tscls[3] = self.tHef(self.mass0, mc1, self.zpars[2])
            # 中等质量恒星
            else:
                self.lums[5] = self.lHef() * self.lums[4]
                self.tscls[3] = self.tHef(self.mass0, 1, self.zpars[2]) * self.tscls[1]
        # 大质量恒星
        else:
            # Note that for M > zpars[3] there is no GB as the star goes from HG -> CHeB -> AGB.
            # So in effect self.tscls[1] refers to the time of Helium ignition and not the BGB.
            self.tscls[2] = self.tscls[1]
            # 这里由于是大质量恒星, 因此氦燃烧时间与核质量无关，可为任意值(此处为1)
            self.tscls[3] = self.tHef(self.mass0, 1, self.zpars[2]) * self.tscls[1]
            # This now represents the luminosity at the end of CHeB, ie. BAGB
            self.lums[5] = self.lums[7]  # 【疑问】为什么对于大质量恒星, 氦燃烧的光度等于BAGB的光度？
            # We set lums[3] to be the luminosity at the end of the HG
            self.lums[3] = self.lums[4]

        # 设置巨星分支底部(bgb)的核质量
        if self.mass0 <= self.zpars[2]:
            self.GB[9] = self.lum_to_mc_gb(self.lums[3])
        elif self.mass0 <= self.zpars[3]:
            self.GB[9] = self.mc_bgb(self.mass0)
        else:
            self.GB[9] = self.mc_bgb(self.mass0, stage='HeI')

        # 设置巨星氦点燃时的核质量
        if self.mass0 <= self.zpars[2]:
            self.GB[10] = self.lum_to_mc_gb(self.lums[4])
        else:
            self.GB[10] = self.mc_bgb(self.mass0, stage='HeI')

        # EAGB 时标参数
        tbagb = self.tscls[2] + self.tscls[3]
        self.tscls[7] = tbagb + (1 / ((self.GB[5] - 1) * self.GB[8] * self.GB[4])) * (
                    (self.GB[4] / self.lums[7]) ** ((self.GB[5] - 1) / self.GB[5]))
        self.tscls[9] = self.tscls[7] - (self.tscls[7] - tbagb) * (
                    (self.lums[7] / self.lums[6]) ** ((self.GB[5] - 1) / self.GB[5]))
        self.tscls[8] = self.tscls[9] + (1 / ((self.GB[6] - 1) * self.GB[8] * self.GB[3])) * (
                    (self.GB[3] / self.lums[6]) ** ((self.GB[6] - 1) / self.GB[6]))

        # 设置渐近巨星分支底部(bagb)的核质量
        self.GB[11] = self.mc_bagb(self.mass0)

        # Now to find Ltp and ttp using Mc,He,tp
        mcbagb = self.mc_bagb(self.mass0)
        mc1 = mcbagb
        # The star undergoes dredge-up at Ltp causing a decrease in Mc, He
        if 0.8 <= mc1 < 2.25:
            mc1 = 0.44 * mc1 + 0.448
        self.lums[8] = self.mc_to_lum_gb(mc1, self.GB)
        if mc1 <= self.GB[7]:
            self.tscls[13] = self.tscls[7] - (1 / ((self.GB[5] - 1) * self.GB[8] * self.GB[4])) * (
                        mc1 ** (1 - self.GB[5]))
        else:
            self.tscls[13] = self.tscls[8] - (1 / ((self.GB[6] - 1) * self.GB[8] * self.GB[3])) * (
                        mc1 ** (1 - self.GB[6]))

        # TPAGB 时标参数
        if mc1 <= self.GB[7]:
            self.tscls[10] = self.tscls[13] + (1 / ((self.GB[5] - 1) * self.GB[2] * self.GB[4])) * (
                        (self.GB[4] / self.lums[8]) ** ((self.GB[5] - 1) / self.GB[5]))
            self.tscls[12] = self.tscls[10] - (self.tscls[10] - self.tscls[13]) * (
                        (self.lums[8] / self.lums[6]) ** ((self.GB[5] - 1) / self.GB[5]))
            self.tscls[11] = self.tscls[12] + (1 / ((self.GB[6] - 1) * self.GB[2] * self.GB[3])) * (
                        (self.GB[3] / self.lums[6]) ** ((self.GB[6] - 1) / self.GB[6]))
        else:
            self.tscls[10] = self.tscls[7]
            self.tscls[12] = self.tscls[9]
            self.tscls[11] = self.tscls[13] + (1 / ((self.GB[6] - 1) * self.GB[2] * self.GB[3])) * (
                        (self.GB[3] / self.lums[8]) ** ((self.GB[6] - 1) / self.GB[6]))

        # Get an idea of when Mc,C = Mc,C,max on the AGB
        tau = self.tscls[2] + self.tscls[3]
        mc2 = self.mcgbtf(tau, self.GB[8], self.GB, self.tscls[7], self.tscls[8], self.tscls[9])
        mcmax = max(max(M_ch, 0.773 * mcbagb - 0.35), 1.05 * mc2)
        if mcmax <= mc1:
            if mcmax <= self.GB[7]:
                self.tscls[14] = self.tscls[7] - (1 / ((self.GB[5] - 1) * self.GB[8] * self.GB[4])) * (
                            mcmax ** (1 - self.GB[5]))
            else:
                self.tscls[14] = self.tscls[8] - (1 / ((self.GB[6] - 1) * self.GB[8] * self.GB[3])) * (
                            mcmax ** (1 - self.GB[6]))
        # Star is on SAGB and we need to increase mcmax if any 3rd dredge-up has occurred.
        else:
            Lambda = min(0.9, 0.3 + 0.001 * self.mass0 ** 5)  # 这里的 Lambda 仅为局部变量
            mcmax = (mcmax - Lambda * mc1) / (1 - Lambda)
            if mcmax <= self.GB[7]:
                self.tscls[14] = self.tscls[10] - (1 / ((self.GB[5] - 1) * self.GB[2] * self.GB[4])) * (
                            mcmax ** (1 - self.GB[5]))
            else:
                self.tscls[14] = self.tscls[11] - (1 / ((self.GB[6] - 1) * self.GB[2] * self.GB[3])) * (
                            mcmax ** (1 - self.GB[6]))
        self.tscls[14] = np.maximum(tbagb, self.tscls[14])
        if self.mass0 > 100:
            self.tn = self.tscls[2]
            return

        # 计算核时标: 不考虑进一步的质量损失时, 耗尽核燃料的时间。我们定义 Mc = self.mass 的时间为 Tn, 这也会用于确定所需的时间步长
        # 注意, 当某些恒星达到 Mc = self.mass 之后还会有一个氦星的演化时间, 后者也是一个核燃烧阶段, 但并不包括在 self.tn 内
        if abs(self.mass - mcbagb) < 1e-14 and self.type < 5:
            self.tn = tbagb
        # Note that the only occurence of Mc being double-valued is for stars that have a dredge-up.
        # If self.mass = Mc where Mc could be the value taken from CHeB or from the AGB we need to check the current stellar type.
        else:
            if self.mass > mcbagb or (self.mass >= mc1 and self.type > 4):
                if self.type == 6:
                    Lambda = min(0.9, 0.3 + 0.001 * self.mass0 ** 5)  # 这里的 Lambda 仅为局部变量
                    mc1 = (self.mass - Lambda * mc1) / (1 - Lambda)
                else:
                    mc1 = self.mass
                if mc1 <= self.GB[7]:
                    self.tn = self.tscls[10] - (1 / ((self.GB[5] - 1) * self.GB[2] * self.GB[4])) * (
                                mc1 ** (1 - self.GB[5]))
                else:
                    self.tn = self.tscls[11] - (1 / ((self.GB[6] - 1) * self.GB[2] * self.GB[3])) * (
                                mc1 ** (1 - self.GB[6]))
            else:
                # 大质量恒星
                if self.mass0 > self.zpars[3]:
                    mc1 = self.mc_bgb(self.mass0, stage='HeI')
                    if self.mass <= mc1:
                        self.tn = self.tscls[2]
                    else:
                        self.tn = self.tscls[2] + self.tscls[3] * ((self.mass - mc1) / (mcbagb - mc1))
                # 小质量恒星
                elif self.mass0 <= self.zpars[2]:
                    mc1 = self.lum_to_mc_gb(self.lums[3])
                    mc2 = self.lum_to_mc_gb(self.lums[4])
                    if self.mass <= mc1:
                        self.tn = self.tscls[1]
                    elif self.mass <= mc2:
                        if self.mass <= self.GB[7]:
                            self.tn = self.tscls[4] - (1 / ((self.GB[5] - 1) * self.GB[1] * self.GB[4])) * (
                                        self.mass ** (1 - self.GB[5]))
                        else:
                            self.tn = self.tscls[5] - (1 / ((self.GB[6] - 1) * self.GB[1] * self.GB[3])) * (
                                        self.mass ** (1 - self.GB[6]))
                    else:
                        self.tn = self.tscls[2] + self.tscls[3] * ((self.mass - mc2) / (mcbagb - mc2))
                # 中等质量恒星
                else:
                    mc1 = self.mc_bgb(self.mass0)
                    mc2 = self.mc_bgb(self.mass0, stage='HeI')
                    if self.mass <= mc1:
                        self.tn = self.tscls[1]
                    elif self.mass <= mc2:
                        tgb = self.tscls[2] - self.tscls[1]
                        self.tn = self.tscls[1] + tgb * ((self.mass - mc1) / (mc2 - mc1))
                    else:
                        self.tn = self.tscls[2] + self.tscls[3] * ((self.mass - mc2) / (mcbagb - mc2))
        self.tn = np.minimum(self.tn, self.tscls[14])

    # ------------------------------------------------------------------------------------------------------------------
    #                                        推导氦星的典型时标、特征光度、巨星分支参数
    # ------------------------------------------------------------------------------------------------------------------
    def StellarCal_He_star(self):
        # 估算 He 星的主序时间
        self.tm = self.themsf()
        self.tscls[1] = self.tm
        # He 星在零龄主序和主序末尾的光度
        self.lums[1] = self.lzhef()
        self.lums[2] = self.lums[1] * (1 + 0.45 + max(0.0, 0.85 - 0.08 * self.mass0))
        # 设置 He 星 GB 参数
        self.GB[8] = 8.0e-5
        self.GB[3] = 4.1e4
        self.GB[4] = 5.5e4 / (1 + 0.4 * self.mass0 ** 4)
        self.GB[5] = 5
        self.GB[6] = 3
        self.GB[7] = (self.GB[3] / self.GB[4]) ** (1 / (self.GB[5] - self.GB[6]))
        # Change in slope of giant L-Mc relation
        self.lums[6] = self.GB[4] * self.GB[7] ** self.GB[5]
        # 设置 He 星的 GB 时标(下面的mc1表示HeMS末尾的核质量)
        mc1 = self.lum_to_mc_gb(self.lums[2])
        self.tscls[4] = self.tm + (1 / ((self.GB[5] - 1) * self.GB[8] * self.GB[4])) * mc1 ** (1 - self.GB[5])
        self.tscls[6] = self.tscls[4] - (self.tscls[4] - self.tm) * ((self.GB[7] / mc1) ** (1 - self.GB[5]))
        self.tscls[5] = self.tscls[6] + (1 / ((self.GB[6] - 1) * self.GB[8] * self.GB[3])) * self.GB[7] ** (
                    1 - self.GB[6])
        # 确定氦巨星 CO 核质量达到最大值的时标
        mcmax = min(self.mass, 1.45 * self.mass - 0.31)
        if mcmax <= 0:
            mcmax = self.mass
        mcmax = min(mcmax, max(M_ch, 0.773 * self.mass0 - 0.35))
        if mcmax <= self.GB[7]:
            self.tscls[14] = self.tscls[4] - (1 / ((self.GB[5] - 1) * self.GB[8] * self.GB[4])) * (
                        mcmax ** (1 - self.GB[5]))
        else:
            self.tscls[14] = self.tscls[5] - (1 / ((self.GB[6] - 1) * self.GB[8] * self.GB[3])) * (
                        mcmax ** (1 - self.GB[6]))
        self.tscls[14] = np.maximum(self.tscls[14], self.tm)
        self.tn = self.tscls[14]

    # ------------------------------------------------------------------------------------------------------------------
    #                                        推导致密星的典型时标、特征光度、巨星分支参数
    # ------------------------------------------------------------------------------------------------------------------
    def StellarCal_CO(self):
        self.tm = 1e10
        self.tscls[1] = self.tm
        self.tn = 1e10


    # ------------------------------------------------------------------------------------------------------------------
    #                              确定恒星目前处于哪一个演化阶段, 然后计算光度/半径/质量/核质量
    # ------------------------------------------------------------------------------------------------------------------
    def StellarProp(self):
        # 为了细致演化, 每个步长内恒星类型只变动一次, 即避免在一个步长内完成类似演化: EAGB → He star(after CE) → BH

        # 计算恒星的光度/半径/核质量
        if self.type <= 6:
            self.StellarProp_H_star()
        elif self.type <= 9:
            self.StellarProp_He_star()
        elif self.type <= 12:
            self.StellarProp_WD()
        elif self.type == 13:
            self.StellarProp_NS()
        elif self.type == 14:
            self.StellarProp_BH()
        elif self.type == 15:
            self.StellarProp_Massless_remnant()
        else:
            raise ValueError("Unsupported stellar type. Expected an integer type in the range 0..15.")

        # 计算恒星核的光度/半径
        self.StellarProp_core()

        # 考虑包层显著减少（星风、物质转移）情况下的光度/半径扰动(主序星除外)
        self.StellarProp_perturb()

        # 估算对流包层的质量、半径, 以及包层的 gyration radius
        self.StellarProp_convective_envelope()

    # ------------------------------------------------------------------------------------------------------------------
    #                                         计算富氢恒星光度/半径/核质量
    # ------------------------------------------------------------------------------------------------------------------
    def StellarProp_H_star(self):
        # 一些预设值
        mlp = 12.0
        tbagb = self.tscls[2] + self.tscls[3]
        rzams = self.rzamsf()
        rtms = self.rtmsf()

        # 主序和赫氏空隙两个阶段
        if self.age < self.tscls[1]:
            self.rg = self.rgbf(self.mass, self.lums[3])
            # 主序阶段(通常认为这个阶段核质量为 0)
            if self.age < self.tm:
                self.M_core = 0.
                tau = self.age / self.tm
                thook = self.thook_div_tBGB() * self.tscls[1]
                epsilon = 0.01
                tau1 = min(1.0, self.age / thook)
                tau2 = max(0.0, min(1.0, (self.age - (1.0 - epsilon) * thook) / (epsilon * thook)))

                # 计算主序阶段光度
                delta_L = self.lpertf()
                dtau = tau1 ** 2 - tau2 ** 2
                alpha_L = self.lalphaf()
                beta_L = self.lbetaf()
                eta = self.lnetaf()
                lx = np.log10(self.lums[2] / self.lums[1])
                xx = alpha_L * tau + beta_L * tau ** eta + (lx - alpha_L - beta_L) * tau ** 2 - delta_L * dtau
                self.L = self.lums[1] * 10 ** xx

                # 计算主序阶段半径
                delta_R = self.rpertf()
                dtau = tau1 ** 3 - tau2 ** 3
                alpha_R = self.ralphaf()
                beta_R = self.rbetaf()
                gamma = self.rgammaf()
                rx = np.log10(rtms / rzams)
                xx = alpha_R * tau + beta_R * tau ** 10 + gamma * tau ** 40 + (
                        rx - alpha_R - beta_R - gamma) * tau ** 3 - delta_R * dtau
                self.R = rzams * 10.0 ** xx

                # This following is given by Chris for low mass MS stars which will be substantially degenerate.
                # We need the Hydrogen abundance X, which we calculate according to X = 0.76 - 3*Z,
                # the helium abundance Y, is calculated according to Y = 0.24 + 2*Z
                if self.mass0 < self.zpars[1] - 0.3:
                    self.type = 0
                    self.R = max(self.R, 0.0258 * ((1 + self.zpars[11]) ** (5 / 3)) * (self.mass0 ** (-1 / 3)))
                else:
                    self.type = 1

            # 赫氏空隙阶段
            else:
                # 计算核质量
                if self.mass0 <= self.zpars[2]:
                    mcEHG = self.lum_to_mc_gb(self.lums[3])
                elif self.mass0 <= self.zpars[3]:
                    mcEHG = self.mc_bgb(self.mass0)
                else:
                    mcEHG = self.mc_bgb(self.mass0, stage='HeI')
                rho = self.mctmsf()
                tau = (self.age - self.tm) / (self.tscls[1] - self.tm)
                mc_new = ((1.0 - tau) * rho + tau) * mcEHG
                self.M_core = max(self.M_core, mc_new)
                # 检验核质量是否达到当前的总质量(如果达到，则说明包层已被剥离，氦核根据是否简并分别演化为氦主序或氦白矮星)
                if self.mass - self.M_core <= 1e-10:
                    # 非简并氦核, 且当前质量大于He点燃的临界值, 则演变为零龄 HeMS
                    if self.mass0 > self.zpars[2] and self.mass > self.zpars[10]:
                        self.type = 7
                        self.age = 0.
                        self.mass0 = self.mass
                        self.StellarCal()
                        self.StellarProp_He_star(initialize=True)
                    # 简并氦核, 则演变为零龄 HeWD
                    else:
                        self.type = 10
                        self.age = 0.
                        self.StellarCal()
                        self.StellarProp_WD(initialize=True)
                else:
                    self.type = 2
                    # 计算赫氏空隙阶段光度
                    self.L = self.lums[2] * (self.lums[3] / self.lums[2]) ** tau

                    # 计算赫氏空隙阶段半径
                    # 中低质量的 HG 末尾在 BGB 处
                    if self.mass0 <= self.zpars[3]:
                        rx = self.rg
                    # 大质量的 HG 末尾在 He 点燃时(at Rmin)
                    else:
                        # 首先算一下 blue loop 阶段的最小半径
                        rmin = self.rminf(self.mass0)
                        # 然后算一下 He 点燃时的半径
                        ry = self.ragbf(self.mass, self.lums[4], self.zpars[2])
                        rx = min(rmin, ry)
                        if self.mass0 <= mlp:
                            texp = np.log(self.mass0 / mlp) / np.log(self.zpars[3] / mlp)
                            rx = self.rg
                            rx = rmin * (rx / rmin) ** texp
                        tau2 = self.tblf()
                        if tau2 < tiny:
                            rx = ry
                    self.R = rtms * (rx / rtms) ** tau

        # 巨星分支
        elif self.age < self.tscls[2]:
            self.type = 3
            # 计算光度和半径
            self.L = self.lgbtf(self.GB[1])
            self.R = self.rgbf(self.mass, self.L)
            self.rg = self.R
            # 计算核质量(对于核是否简并有不同的核质量公式)
            # 核简并时，核的质量在GB上持续增加
            if self.mass0 <= self.zpars[2]:
                self.M_core = self.lum_to_mc_gb(self.L)
            # 非简并核的质量在GB阶段只会轻微的增加
            else:
                tau = (self.age - self.tscls[1]) / (self.tscls[2] - self.tscls[1])
                mc_bgb = self.mc_bgb(self.mass0)
                mc_hei = self.mc_bgb(self.mass0, stage='HeI')
                self.M_core = mc_bgb + (mc_hei - mc_bgb) * tau
            # 检验核质量是否达到当前的总质量
            if self.mass - self.M_core <= 1e-10:
                # 非简并氦核, 且当前质量大于He点燃的临界值, 则演变为零龄 HeMS
                if self.mass0 > self.zpars[2] and self.mass > self.zpars[10]:
                    self.type = 7
                    self.age = 0.
                    self.mass0 = self.mass
                    self.StellarCal()
                    self.StellarProp_He_star(initialize=True)
                # 简并氦核, 则演变为零龄 HeWD
                else:
                    self.type = 10
                    self.age = 0.
                    self.StellarCal()
                    self.StellarProp_WD(initialize=True)

        # 水平分支
        elif self.age < tbagb:
            if self.type == 3 and self.mass0 <= self.zpars[2]:
                self.mass0 = self.mass  # 这里为什么改变初始质量？不懂！
                self.StellarCal()
                self.age = self.tscls[2]

            # 计算核质量
            if self.mass0 <= self.zpars[2]:
                mchei = self.lum_to_mc_gb(self.lums[4])
            else:
                mchei = self.mc_bgb(self.mass0, stage='HeI')
            tau = (self.age - self.tscls[2]) / self.tscls[3]
            self.M_core = mchei + (self.mc_bagb(self.mass0) - mchei) * tau

            # 低质量恒星
            if self.mass0 <= self.zpars[2]:
                lx = self.lums[5]
                ly = self.lums[7]
                rx = self.rzahbf(self.mass, self.M_core, self.zpars[2])
                rg = self.rgbf(self.mass, lx)
                rmin = rg * self.zpars[13] ** (self.mass0 / self.zpars[2])
                texp = min(max(0.4, rmin / rx), 2.5)
                ry = self.ragbf(self.mass, ly, self.zpars[2])
                if rmin < rx:
                    taul = (np.log(rx / rmin)) ** (1 / 3)
                else:
                    rmin = rx
                    taul = 0.0
                tauh = (np.log(ry / rmin)) ** (1 / 3)
                tau2 = taul * (tau - 1.0) + tauh * tau
                self.R = rmin * np.exp(abs(tau2) ** 3)
                self.rg = rg + tau * (ry - rg)
                self.L = lx * (ly / lx) ** (tau ** texp)

            # 大质量恒星, 氦点燃发生在 HG 上的最小半径 (Rmin) 处
            # CHeB consists of a blue phase (before tloop) and a RG phase (after tloop).
            elif self.mass0 > self.zpars[3]:
                tau2 = self.tblf()
                tloop = self.tscls[2] + tau2 * self.tscls[3]
                rmin = self.rminf(self.mass0)
                rg = self.rgbf(self.mass, self.lums[4])
                rx = self.ragbf(self.mass, self.lums[4], self.zpars[2])
                rmin = min(rmin, rx)
                if self.mass0 <= mlp:
                    texp = np.log(self.mass0 / mlp) / np.log(self.zpars[3] / mlp)
                    rx = rg
                    rx = rmin * (rx / rmin) ** texp
                else:
                    rx = rmin
                texp = min(max(0.4, rmin / rx), 2.5)
                self.L = self.lums[4] * (self.lums[7] / self.lums[4]) ** (tau ** texp)
                if self.age < tloop:
                    ly = self.lums[4] * (self.lums[7] / self.lums[4]) ** (tau2 ** texp)
                    ry = self.ragbf(self.mass, ly, self.zpars[2])
                    taul = 0.
                    if abs(rmin - rx) > tiny:
                        taul = (np.log(rx / rmin)) ** (1 / 3)
                    tauh = 0.
                    if ry > rmin:
                        tauh = (np.log(ry / rmin)) ** (1 / 3)
                    tau = (self.age - self.tscls[2]) / (tau2 * self.tscls[3])
                    tau2 = taul * (tau - 1.0) + tauh * tau
                    self.R = rmin * np.exp(abs(tau2) ** 3)
                    self.rg = rg + tau * (ry - rg)
                else:
                    self.R = self.ragbf(self.mass, self.L, self.zpars[2])
                    self.rg = self.R

            # 中等质量恒星, CHeB consists of a RG phase (before tloop) and a blue loop (after tloop).
            else:
                tau2 = 1.0 - self.tblf()
                tloop = self.tscls[2] + tau2 * self.tscls[3]
                if self.age < tloop:
                    tau = (tloop - self.age) / (tau2 * self.tscls[3])
                    self.L = self.lums[5] * (self.lums[4] / self.lums[5]) ** (tau ** 3)
                    self.R = self.rgbf(self.mass, self.L)
                    self.rg = self.R
                else:
                    lx = self.lums[5]
                    ly = self.lums[7]
                    rx = self.rgbf(self.mass, lx)
                    rmin = self.rminf(self.mass)
                    texp = min(max(0.4, rmin / rx), 2.5)
                    ry = self.ragbf(self.mass, ly, self.zpars[2])
                    if rmin < rx:
                        taul = (np.log(rx / rmin)) ** (1 / 3)
                    else:
                        rmin = rx
                        taul = 0.

                    tauh = (np.log(ry / rmin)) ** (1 / 3)
                    tau = (self.age - tloop) / (self.tscls[3] - (tloop - self.tscls[2]))
                    tau2 = taul * (tau - 1.0) + tauh * tau
                    self.R = rmin * np.exp(abs(tau2) ** 3)
                    self.rg = rx + tau * (ry - rx)
                    self.L = lx * (ly / lx) ** (tau ** texp)

            # 检验核质量是否达到当前的总质量
            if self.mass - self.M_core <= 1e-10:
                self.type = 7
                tau = (self.age - self.tscls[2]) / self.tscls[3]
                # 把氦星的初始质量近似为当前的核质量, 因为后者的实际值无法计算
                self.mass0 = self.mass
                self.StellarCal()
                self.age = tau * self.tm
                self.StellarProp_He_star(initialize=True)
            else:
                self.type = 4

        # 渐近巨星分支
        else:
            # 以下的 mc_CO 表示CO核的质量, 部分情况也表示ONe核的质量
            mcbagb = self.mc_bagb(self.mass0)  # BAGB时的核质量(He + CO)
            mc_CO_bagb = self.mcgbtf(tbagb, self.GB[8], self.GB, self.tscls[7], self.tscls[8],
                                     self.tscls[9])  # BAGB时的CO核质量
            # 根据mcbagb质量不同, 超新星爆发有不同的临界质量
            # 对于简并碳氧核, 超新星爆发的核质量极限是Mch
            if mcbagb < 1.83:
                mc_max_SN = M_ch
            # 对于半简并碳氧核, 在 M_CO = 1.08M_sun 时会发生非中心点燃生成简并ONeMg核, 而ONe核发生ECSN爆发的质量极限是1.38M_sun
            elif mcbagb < 2.25:
                mc_max_SN = M_ECSN
            # 对于非简并碳氧核, 可以一直燃烧到Fe核形成, SN爆炸的质量极限根据mcbagb确定
            else:
                mc_max_SN = 0.773 * mcbagb - 0.35

            # CO核/ONe核的质量有两个上限: SN爆炸极限质量和当前恒星总质量(后者情况, 包层被剥离, 核未达到SN极限, 只能变成CO/ONe WD)
            # CO核/ONe核的质量上限不应该受到mcbagb的限制, 因为只要有包层, H → He → CO就会一直发生, 即CO核质量持续增加
            # 我并不清楚下面的1.05 * mc_CO_bagb原因, 尽管这某种程度上更加的物理, 因为SN时的CO核质量一定大于BAGB时的CO核质量, 可能是
            # 由于0.773 * mcbagb - 0.35这个公式拟合的不够好所以添加了这个补充条件
            mcmax = max(mc_max_SN, 1.05 * mc_CO_bagb)

            # EAGB 阶段, Mc = Mc_He + Mc_CO = Mc_bagb(常数), 而Mc_CO 随时间不断增长, 直到全部的He核转为CO核, EAGB结束
            # 对于0.8 < Mc_bagb < 2.25的恒星, 会有一个second dredge-up阶段, 因此在EAGB末尾的CO核质量到不了Mc_bagb
            if self.age < self.tscls[13]:
                self.type = 5
                self.M_core = mcbagb
                self.M_co_core = self.mcgbtf(self.age, self.GB[8], self.GB, self.tscls[7], self.tscls[8], self.tscls[9])
                # 相应光度根据 L-mc_CO 关系变化
                self.L = self.mc_to_lum_gb(self.M_co_core, self.GB)
                # 如果当前核质量大于恒星总质量, 说明包层已经损失, 但由于氦核没有全部燃烧完, 因此成为post-HeMS 裸氦星
                if self.mass - self.M_core <= 1e-10:
                    self.type = 9
                    self.mass0 = self.M_core
                    self.mass = self.M_core
                    self.StellarCal()
                    if self.M_co_core <= self.GB[7]:
                        self.age = self.tscls[4] - (1.0 / ((self.GB[5] - 1.0) * self.GB[8] * self.GB[4])) * (
                                    self.M_co_core ** (1.0 - self.GB[5]))
                    else:
                        self.age = self.tscls[5] - (1.0 / ((self.GB[6] - 1.0) * self.GB[8] * self.GB[3])) * (
                                    self.M_co_core ** (1.0 - self.GB[6]))
                    self.age = max(self.age, self.tm)
                    self.StellarProp_He_star(initialize=True)
                    return

            # TPAGB 阶段, Mc = Mc_CO, 如果能达到 Mcmax, 则根据此时的 Mc 演化成不同的恒星类型
            else:
                self.type = 6
                # TPAGB开始时的CO核质量
                mc_CO_1 = self.mcgbtf(self.tscls[13], self.GB[2], self.GB, self.tscls[10], self.tscls[11],
                                      self.tscls[12])
                # TPAGB开始后没有三次挖掘时的CO核质量
                self.M_co_core = self.mcgbtf(self.age, self.GB[2], self.GB, self.tscls[10], self.tscls[11],
                                             self.tscls[12])
                self.L = self.mc_to_lum_gb(self.M_co_core, self.GB)
                # 由于三次挖掘(3rd Dredge-up), Mc的增长变缓
                f_lambda = min(0.9, 0.3 + 0.001 * self.mass0 ** 5)
                self.M_co_core = self.M_co_core - f_lambda * (self.M_co_core - mc_CO_1)
                self.M_core = self.M_co_core

                # 如果当前核质量等于恒星总质量, 说明包层已经损失, 由于只剩下了CO/ONe核, 根据简并与否判定最终产物(详见处理氦星时的情况)
                if self.mass - self.M_core <= 1e-10:
                    self.age = 0
                    self.M_core = min(self.mass, mcmax)
                    # 简并CO核质量未达到 mch , 只能变为CO白矮星
                    if mcbagb < 1.83:
                        self.type = 11
                        self.mass = self.M_core
                        self.StellarCal()
                        self.StellarProp_WD(initialize=True)
                    # 半简并的CO核(非中心)点燃形成简并的ONe核, 简并ONe核质量未能达到电子俘获超新星临界质量 Mecs, 只能成为ONe白矮星
                    elif mcbagb < 2.25:
                        self.type = 12
                        self.mass = self.M_core
                        self.StellarCal()
                        self.StellarProp_WD(initialize=True)
                    # 非简并的CO核发生超新星爆炸(这种大质量的恒星一般在进入TPAGB之前就发生了SN, 所以下面这个分支大概率用不到)
                    else:
                        self.SN_remnant(mcbagb)
                        self.StellarCal()
                        if self.type == 13:
                            self.StellarProp_NS(initialize=True)
                        else:
                            self.StellarProp_BH()
                    return

            # 检验CO/ONe核质量是否超过超新星爆炸极限质量
            # 大质量恒星会在EAGB发生SN, 此时核质量=He核质量, 因此需用CO核质量和临界值比较, 然后将临界值作为核质量, 方便SN演化
            if mcmax - self.M_co_core <= 1e-10:
                self.age = 0.0
                self.M_core = mcmax
                # 简并CO核质量达到 mch 后, 星体坍缩引发Ia超新星爆炸后不会留下恒星遗迹
                if mcbagb < 1.83:
                    self.type = 15
                    self.event = 'Ia'
                    self.StellarCal()
                    self.StellarProp_Massless_remnant()
                # 半简并的碳氧核(非中心)点燃形成简并的氧氖核, 核质量达到 M_ECSN 后经电子俘获型超新星爆发, 留下中子星
                elif mcbagb < 2.25:
                    self.type = 13
                    self.mass = 1.3
                    self.event = 'ECSN'
                    self.StellarCal()
                    self.StellarProp_NS(initialize=True)
                # 非简并的CO核在中心点燃, 最终重元素燃烧生成铁核, 经历铁核坍缩后发生超新星爆炸, 留下中子星或黑洞
                else:
                    self.SN_remnant(mcbagb)
                    self.StellarCal()
                    if self.type == 13:
                        self.StellarProp_NS(initialize=True)
                    else:
                        self.StellarProp_BH()
            # 计算半径
            else:
                self.R = self.ragbf(self.mass, self.L, self.zpars[2])
                self.rg = self.R

    # ------------------------------------------------------------------------------------------------------------------
    #                                            计算氦星的光度/半径/核质量
    # ------------------------------------------------------------------------------------------------------------------
    def StellarProp_He_star(self, initialize=False):
        lzams = self.lzhef()
        # 这里计算半径用的是当前质量
        rzams = self.rzhef(self.mass)
        # Main Sequence
        if self.age < self.tm:
            self.type = 7
            tau = self.age / self.tm
            self.L = lzams * (1 + 0.45 * tau + max(0, 0.85 - 0.08 * self.mass0) * tau ** 2)
            self.R = rzams * (1 + max(0, 0.4 - 0.22 * np.log10(self.mass)) * (tau - tau ** 6))
            self.rg = rzams  # 这个变量可能之后包层演化程序中会用到
            # Star has no core mass and hence no memory of its past which is
            # why we subject mass and mt to mass loss for this phase.
            self.M_core = 0.0
            if self.mass < self.zpars[10]:
                self.type = 10
        # Helium Shell Burning
        else:
            self.type = 8
            self.L = self.lgbtf(self.GB[8])
            self.R = self.rhehgf(self.mass, self.L, rzams, self.lums[2])
            self.rg = self.rhegbf(self.L)
            if self.R >= self.rg:
                self.type = 9
                self.R = self.rg
            self.M_core = self.lum_to_mc_gb(self.L)

            # 第一种情况, 氦星包层完全被剥离, 简并CO核/简并ONe核演变成白矮星, 非简并CO核触发超新星爆炸
            # 如果He星的质量小于0.7M_sun, He包层无法全部转化为CO核, 因此对小质量He星的CO核质量上限进行限制
            mcmax_1 = min(self.mass, 1.45 * self.mass - 0.31)
            # 第二种情况, 包层还在, 但CO核的质量已经达到超新星爆炸临界值, 如果初始质量小于1.83M_sun, 则为简并CO核, 最大核质量上限为Mch;
            # 如果初始质量范围是1.83-2.25M_sun, 则为简并ONe核, 最大核质量上限为M_ECSN;如果初始质量>2.25M_sun, 最大核质量根据初始质量决定
            mcmax_2 = M_ch if self.mass0 < 1.83 else M_ECSN if self.mass0 < 2.25 else 0.773 * self.mass0 - 0.35
            mcmax = min(mcmax_1, mcmax_2)

            # 简并CO核, 根据核质量变成CO白矮星或引发Ia超新星
            if self.mass0 < 1.83:
                if mcmax - self.M_core < 1e-10 and not initialize:
                    self.age = 0
                    self.M_core = mcmax
                    if mcmax < mcmax_2:
                        self.type = 11
                        self.mass = self.M_core
                        self.StellarCal()
                        self.StellarProp_WD(initialize=True)
                    else:
                        self.type = 15
                        self.event = 'Ia'
                        self.StellarCal()
                        self.StellarProp_Massless_remnant()
            # 简并的ONe核, 根据核质量变成ONe白矮星或引发ECSN留下中子星
            elif self.mass0 < 2.25:
                if mcmax - self.M_core < 1e-10 and not initialize:
                    self.age = 0
                    self.M_core = mcmax
                    if mcmax < mcmax_2:
                        self.type = 12
                        self.mass = self.M_core
                        self.StellarCal()
                        self.StellarProp_WD(initialize=True)
                    else:
                        self.type = 13
                        self.mass = 1.3
                        self.event = 'ECSN'
                        self.StellarCal()
                        self.StellarProp_NS(initialize=True)
            # 非简并的CO核, 如果包层被剥离后还没达到SN爆炸临界值, 热核会冷却由非简并 → 简并, 根据热核质量确定最终结果(这里尚待商榷)
            else:
                # print(self.step, self.type, self.mass0, self.mass, mcmax, self.M_core)
                if mcmax - self.M_core < 1e-10 and not initialize:
                    self.age = 0
                    self.M_core = mcmax
                    if mcmax < mcmax_2:
                        if self.M_core < 1.08:
                            self.type = 11
                            self.mass = self.M_core
                            self.StellarCal()
                            self.StellarProp_WD(initialize=True)
                        elif self.M_core < M_ECSN:
                            self.type = 12
                            self.mass = self.M_core
                            self.StellarCal()
                            self.StellarProp_WD(initialize=True)
                        elif self.M_core < M_ch:
                            self.type = 13
                            self.mass = 1.3
                            self.event = 'ECSN'
                            self.StellarCal()
                            self.StellarProp_NS(initialize=True)
                        else:
                            self.SN_remnant(mcbagb=self.mass)
                            self.StellarCal()
                            if self.type == 13:
                                self.StellarProp_NS(initialize=True)
                            else:
                                self.StellarProp_BH()
                    else:
                        self.SN_remnant(mcbagb=self.mass)
                        self.StellarCal()
                        if self.type == 13:
                            self.StellarProp_NS(initialize=True)
                        else:
                            self.StellarProp_BH()

    # ------------------------------------------------------------------------------------------------------------------
    #                                            计算白矮星的光度/半径/核质量
    # ------------------------------------------------------------------------------------------------------------------
    def StellarProp_WD(self, initialize=False):
        self.M_core = self.mass

        # 设置初始质量, 方便后续计算: 当COWD吸积超过0.15M_sun富氦物质, 发生Ia SN
        if initialize:
            self.mass0 = self.mass

        if self.type == 12:
            if self.mass >= M_ECSN and not initialize:
                self.type = 13
                self.age = 0.0
                self.mass = 1.3
                self.event = 'AIC'
                self.StellarCal()
                self.StellarProp_NS(initialize=True)
                return
        else:
            if self.mass >= M_ch and not initialize:
                self.type = 15
                self.event = 'Ia'
                self.StellarCal()
                self.StellarProp_Massless_remnant()
                return

        xx = ahe if self.type == 10 else aco

        # modified-Mestel cooling  (未使用)
        if WD_flag:
            if self.age < 9000:
                self.L = 300 * self.mass * self.zpars[14] / (xx * (self.age + 0.1)) ** 1.18
            else:
                fac = (9000.1 * xx) ** 5.3
                self.L = 300 * fac * self.mass * self.zpars[14] / (xx * (self.age + 0.1)) ** 6.48
        # Mestel cooling
        else:
            self.L = 635 * self.mass * self.zpars[14] / (xx * (self.age + 0.1)) ** 1.4

        # if self.mass == 0:
        #     print(self.step, self.time)
        #     self.mass = 0.5
        self.R = max(1e6 / R_sun, 0.0115 * np.sqrt((M_ch / self.mass) ** (2 / 3) - (self.mass / M_ch) ** (2 / 3)))
        self.R = min(0.1, self.R)
        if self.mass < 0.0005:
            self.R = 0.09
        if self.mass < 0.000005:
            self.R = 0.009

    # ------------------------------------------------------------------------------------------------------------------
    #                                            计算中子星的光度/半径/核质量
    # ------------------------------------------------------------------------------------------------------------------
    def StellarProp_NS(self, initialize=False):
        self.M_core = self.mass

        # AIC黑洞
        if self.mass > M_ns_max and not initialize:
            self.type = 14
            self.age = 0.0
            self.event = 'AIC'
        else:
            self.L = 0.02 * self.mass ** (2 / 3) / (max(self.age, 0.1)) ** 2
            self.R = 1.4e-5

    # ------------------------------------------------------------------------------------------------------------------
    #                                            计算黑洞的光度/半径/核质量
    # ------------------------------------------------------------------------------------------------------------------
    def StellarProp_BH(self):
        self.M_core = self.mass
        self.L = 1.0e-10
        self.R = 4.24e-6 * self.mass

    # ------------------------------------------------------------------------------------------------------------------
    #                                            无质量恒星(Ia、合并等情况)
    # ------------------------------------------------------------------------------------------------------------------
    def StellarProp_Massless_remnant(self):
        self.mass = 0.0
        self.M_core = 0.0
        self.L = 1e-10
        self.R = 1e-10
        self.R_mt = 1e-10
        self.age = 0.0

    # ------------------------------------------------------------------------------------------------------------------
    #                                               计算核半径、核光度
    # ------------------------------------------------------------------------------------------------------------------
    def StellarProp_core(self):
        # 主序阶段
        if self.type <= 1 or self.type == 7:
            self.R_core = 0.
            self.L_core = 0.
        # 赫氏空隙/巨星阶段
        elif 2 <= self.type <= 3:
            # 非简并的氦核
            if self.mass0 > self.zpars[2]:
                self.R_core = self.rzhef(self.M_core)
                self.L_core = self.lzhef(self.M_core)
            # 简并氦核
            else:
                self.R_core = 5 * 0.0115 * np.sqrt(
                    max(1.48204e-6, (M_ch / self.M_core) ** (2 / 3) - (self.M_core / M_ch) ** (2 / 3)))
                if WD_flag:
                    self.L_core = 300.0 * self.M_core * self.zpars[14] / ((ahe * 0.1) ** 1.18)
                else:
                    self.L_core = 635.0 * self.M_core * self.zpars[14] / ((ahe * 0.1) ** 1.4)
        # 水平分支
        elif self.type == 4:
            tau = (self.age - self.tscls[2]) / self.tscls[3]
            self.R_core = self.rzhef(self.M_core) * (
                    1.0 + max(0.0, 0.4 - 0.22 * np.log10(self.M_core)) * (tau - tau ** 6))
            self.L_core = self.lzhef(self.M_core) * (1.0 + 0.45 * tau + max(0.0, 0.85 - 0.08 * self.M_core) * tau ** 2)
        # EAGB 阶段
        elif self.type == 5:
            tbagb = self.tscls[2] + self.tscls[3]
            tau = 3.0 * (self.age - tbagb) / (self.tn - tbagb) if self.tn > tbagb else 0
            # 保存之前的属性
            type_temp, mass0_temp, mass_temp = self.type, self.mass0, self.mass

            # 把此时的核当作是一个氦巨星, 计算核的半径和光度
            self.type, self.mass0, self.mass = 9, self.M_core, self.M_core
            self.StellarCal()
            lc = self.mc_to_lum_gb(self.M_co_core, self.GB)
            lc = self.lums[2] * (lc / self.lums[2]) ** tau if tau < 1 else lc
            rc = self.rzhef(self.M_core)
            self.R_core = min(self.rhehgf(self.M_core, lc, rc, self.lums[2]), self.rhegbf(lc))
            self.L_core = lc

            # 恢复恒星本身类型对应的特征光度/时标
            self.type, self.mass0, self.mass = type_temp, mass0_temp, mass_temp
            self.StellarCal()
        # TPAGB/HeHG/HeGB
        elif self.type == 6 or 8 <= self.type <= 9:
            self.R_core = 5 * 0.0115 * np.sqrt(
                max(1.48204e-6, (M_ch / self.M_core) ** (2 / 3) - (self.M_core / M_ch) ** (2 / 3)))
            if WD_flag:
                self.L_core = 300 * self.M_core * self.zpars[14] / ((aco * 0.1) ** 1.18)
            else:
                self.L_core = 635 * self.M_core * self.zpars[14] / ((aco * 0.1) ** 1.4)
        # 致密星
        else:
            self.R_core = self.R
            self.L_core = 0

    # ------------------------------------------------------------------------------------------------------------------
    #                        对于包层显著减少（星风、物质转移）的情况, 存在光度/半径扰动(主序星除外)
    # ------------------------------------------------------------------------------------------------------------------
    def StellarProp_perturb(self):
        if 2 <= self.type <= 9 and self.type != 7:
            if self.type >= 8:
                mcmax = min(self.mass, 1.45 * self.mass - 0.31)
                mu = ((mcmax - self.M_core) / mcmax) * 5.0
            else:
                kap = -0.5
                lum0 = 7e4
                mu = ((self.mass - self.M_core) / self.mass) * min(5.0, max(1.2, (self.L / lum0) ** kap))

            if mu < 1.0:
                # 光度的扰动
                b = 0.002 * max(1, 2.5 / self.mass)
                s = (1 + b ** 3) * ((mu / b) ** 3) / (1 + (mu / b) ** 3)
                self.L = self.L_core * (self.L / self.L_core) ** s
                # 半径的扰动
                if self.R <= self.R_core or mu <= 0:
                    rpert = 0.0
                else:
                    c = 0.006 * max(1, 2.5 / self.mass)
                    q = np.log(self.R / self.R_core)
                    fac = 0.1 / q
                    facmax = -14 / np.log10(mu)
                    fac = min(fac, facmax)
                    rpert = ((1 + c ** 3) * ((mu / c) ** 3) * (mu ** fac)) / (1 + (mu / c) ** 3)
                self.R = self.R_core * (self.R / self.R_core) ** rpert

    # ------------------------------------------------------------------------------------------------------------------
    #                  Calculate mass and radius of convective envelope, and envelope gyration radius
    # ------------------------------------------------------------------------------------------------------------------
    def StellarProp_convective_envelope(self):
        if self.type >= 10:
            self.M_conv_env = 1.0e-10
            self.R_conv_env = 1.0e-10
            self.k2 = 0.21
            return

        rzams = self.rzamsf() if self.type <= 6 else self.rzhef(self.mass0)
        rtms = self.rtmsf()  # 【疑问】这里的rtms公式是否对氦星适用

        logm = np.log10(self.mass0)
        A = min(0.81, max(0.68, 0.68 + 0.4 * logm))
        C = max(-2.5, min(-1.5, -2.5 + 5.0 * logm))
        D = -0.1
        E = 0.025

        # 自己加的变量初始化
        tebgb = 0

        # Zero-age and BGB values of k^2.
        k2z = min(0.21, max(0.09 - 0.27 * logm, 0.037 + 0.033 * logm))
        if logm > 1.3:
            k2z = k2z - 0.055 * (logm - 1.3) ** 2
        k2bgb = min(0.15, min(0.147 + 0.03 * logm, 0.162 - 0.04 * logm))

        # Envelope k^2 for giant-like stars; this will be modified for non-giant CHeB stars or small envelope mass below.
        # Formula is fairly accurate for both FGB and AGB stars if M <= 10, and gives reasonable values for higher masses.
        # Mass dependence is on actual rather than ZA mass, expected to work for mass-losing stars (but not tested!).
        # The slightly complex appearance is to insure continuity at the BGB, which depends on the ZA mass.
        if 3 <= self.type <= 6:
            logmt = np.log10(self.mass)
            F = 0.208 + 0.125 * logmt - 0.035 * logmt ** 2
            B = 1e4 * self.mass ** (3.0 / 2.0) / (1.0 + 0.1 * self.mass ** (3.0 / 2.0))
            x = ((self.L - self.lums[3]) / B) ** 2
            y = (F - 0.033 * np.log10(self.lums[3])) / k2bgb - 1.0
            k2g = (F - 0.033 * np.log10(self.L) + 0.4 * x) / (1.0 + y * (self.lums[3] / self.L) + x)
        # Rough fit for HeGB stars...
        elif self.type == 9:
            B = 3e4 * self.mass ** (3.0 / 2.0)
            x = (max(0.0, self.L / B - 0.5)) ** 2
            k2g = (k2bgb + 0.4 * x) / (1.0 + 0.4 * x)
        else:
            k2g = k2bgb

        if self.type <= 2:
            menvg = 0.5
            renvg = 0.65
        # FGB stars still close to the BGB do not yet have a fully developed CE.
        elif self.type == 3 and self.L < 3 * self.lums[3]:
            x = np.minimum(3, self.lums[4] / self.lums[3])
            tau = np.maximum(0, min(1.0, (x - self.L / self.lums[3]) / (x - 1)))
            menvg = 1 - 0.5 * tau ** 2
            renvg = 1 - 0.35 * tau ** 2
        else:
            menvg = 1
            renvg = 1

        # Stars not on the Hayashi track: MS and HG stars, non-giant CHeB stars,
        # HeMS and HeHG stars, as well as giants with very small envelope mass.
        if self.R < self.rg:
            # Envelope k^2 fitted for MS and HG stars.
            # Again, pretty accurate for M <= 10 but less so for larger masses.
            # Note that this represents the whole star on the MS, so there is a discontinuity in stellar k^2
            # between MS and HG - okay for stars with a MS hook but low-mass stars should preferably be continous...
            #
            # For other types of star not on the Hayashi track we use the same fit as for HG stars,
            # this is not very accurate but has the correct qualitative behaviour. For CheB stars
            # this is an overestimate because they appear to have a more centrally concentrated envelope than HG stars.
            if self.type <= 6:
                k2e = (k2z - E) * (self.R / rzams) ** C + E * (self.R / rzams) ** D

            # Rough fit for naked He MS stars.
            elif self.type == 7:
                tau = self.age / self.tm
                k2e = 0.080 - 0.030 * tau
            # Rough fit for HeHG stars.
            # 致密星不会进入当前程序, 之前为elif type<=9
            else:
                k2e = 0.08 * rzams / self.R

            # tauenv measures proximity to the Hayashi track in terms of Teff.
            # If tauenv > 0 then an appreciable convective envelope is present, and k^2 needs to be modified.
            if self.type <= 2:
                teff = np.sqrt(np.sqrt(self.L) / self.R)
                tebgb = np.sqrt(np.sqrt(self.lums[3]) / self.rg)
                tauenv = max(0.0, min(1.0, (tebgb / teff - A) / (1.0 - A)))
            else:
                tauenv = max(0.0, min(1.0, (np.sqrt(self.R / self.rg) - A) / (1.0 - A)))

            if tauenv > 0.0:
                menv = menvg * tauenv ** 5
                renv = renvg * tauenv ** (5.0 / 4.0)
                # Zero-age values for CE mass and radius.
                if self.type <= 1:
                    x = max(0.0, min(1.0, (0.1 - logm) / 0.55))
                    menvz = 0.18 * x + 0.82 * x ** 5
                    renvz = 0.4 * x ** (1.0 / 4.0) + 0.6 * x ** 10
                    y = 2.0 + 8.0 * x
                    # Values for CE mass and radius at start of the HG.
                    tetms = np.sqrt(np.sqrt(self.lums[2]) / rtms)
                    tautms = max(0.0, min(1.0, (tebgb / tetms - A) / (1.0 - A)))
                    menvt = menvg * tautms ** 5
                    renvt = renvg * tautms ** (5.0 / 4.0)
                    # Modified expressions during MS evolution.
                    tau = self.age / self.tm
                    if tautms > 0.0:
                        menv = menvz + tau ** y * menv * (menvt - menvz) / menvt
                        renv = renvz + tau ** y * renv * (renvt - renvz) / renvt
                    else:
                        menv = 0.0
                        renv = 0.0
                    k2e = k2e + tau ** y * tauenv ** 3 * (k2g - k2e)
                else:
                    k2e = k2e + tauenv ** 3 * (k2g - k2e)
            else:
                menv = 0.0
                renv = 0.0
        # All other stars should be true giants.
        else:
            menv = menvg
            renv = renvg
            k2e = k2g

        menv = menv * (self.mass - self.M_core)
        renv = renv * (self.R - self.R_core)
        self.M_conv_env = max(menv, 1e-10)
        self.R_conv_env = max(renv, 1e-10)

        self.k2 = k2e


    # ------------------------------------------------------------------------------------------------------------------
    #                                                  恒星参数拟合公式
    # ------------------------------------------------------------------------------------------------------------------

    # 估算零龄主序光度 Lzams （from Tout et al., 1996, MNRAS, 281, 257）
    def l_zams(self):
        mx = np.sqrt(self.mass0)
        lzams = (self.msp[1] * self.mass0 ** 5 * mx + self.msp[2] * self.mass0 ** 11) / (
                self.msp[3] + self.mass0 ** 3 + self.msp[4] * self.mass0 ** 5 + self.msp[5] * self.mass0 ** 7 +
                self.msp[6] * self.mass0 ** 8 + self.msp[7] * self.mass0 ** 9 * mx)
        return lzams

    # 估算零龄主序半径 Rzams
    def rzamsf(self, m=0):
        mass = self.mass0 if m == 0 else m
        mx = np.sqrt(mass)
        rzams = ((self.msp[8] * mass ** 2 + self.msp[9] * mass ** 6) * mx + self.msp[10] * mass ** 11 + (
                self.msp[11] + self.msp[12] * mx) * mass ** 19) / (self.msp[13] + self.msp[14] * mass ** 2 + (
                self.msp[15] * mass ** 8 + mass ** 18 + self.msp[16] * mass ** 19) * mx)
        return rzams

    # A function to evaluate the lifetime to the BGB or to Helium ignition if no FGB exists. (JH 24/11/97)
    # [已校验] Hurley_2000: equation 5.1(4)
    def tbgbf(self):
        tbgb = (self.msp[17] + self.msp[18] * self.mass0 ** 4 + self.msp[19] * self.mass0 ** (
                    11 / 2) + self.mass0 ** 7) / (
                       self.msp[20] * self.mass0 ** 2 + self.msp[21] * self.mass0 ** 7)
        return tbgb

    # A function to evaluate the derivitive of the lifetime to the BGB
    # (or to Helium ignition if no FGB exists) wrt mass. (JH 24/11/97)
    def tbgbdf(self):
        mx = np.sqrt(self.mass0)
        f = self.msp[17] + self.msp[18] * self.mass0 ** 4 + self.msp[19] * self.mass0 ** 5 * mx + self.mass0 ** 7
        df = 4 * self.msp[18] * self.mass0 ** 3 + 5.5 * self.msp[19] * self.mass0 ** 4 * mx + 7 * self.mass0 ** 6
        g = self.msp[20] * self.mass0 ** 2 + self.msp[21] * self.mass0 ** 7
        dg = 2 * self.msp[20] * self.mass0 + 7 * self.msp[21] * self.mass0 ** 6
        tbgbd = (df * g - f * dg) / (g * g)
        return tbgbd

    # A function to evaluate the derivitive of the lifetime to the BGB
    # (or to Helium ignition if no FGB exists) wrt Z. (JH 14/12/98)
    def tbgdzf(self):
        mx = self.mass0 ** 5 * np.sqrt(self.mass0)
        f = self.msp[17] + self.msp[18] * self.mass0 ** 4 + self.msp[19] * mx + self.mass0 ** 7
        df = self.msp[117] + self.msp[118] * self.mass0 ** 4 + self.msp[119] * mx
        g = self.msp[20] * self.mass0 ** 2 + self.msp[21] * self.mass0 ** 7
        dg = self.msp[120] * self.mass0 ** 2
        tbgdz = (df * g - f * dg) / (g * g)
        return tbgdz

    # A function to evaluate the lifetime to the end of the MS hook as a fraction of the lifetime to the BGB
    # (for those models that have one). Note that this function is only valid for self.mass0 > Mhook.
    # [已校验] Hurley_2000: equation 5.1(7)
    def thook_div_tBGB(self):
        term = 1 - 0.01 * max(self.msp[22] / self.mass0 ** self.msp[23],
                              self.msp[24] + self.msp[25] / self.mass0 ** self.msp[26])
        value = max(0.5, term)
        return value

    # 估算主序末尾的光度
    # [已校验] Hurley_2000: equation 5.1(8)
    def ltmsf(self):
        ltms = (self.msp[27] * self.mass0 ** 3 + self.msp[28] * self.mass0 ** 4 + self.msp[29] * self.mass0 ** (
                    self.msp[32] + 1.8)) / (
                       self.msp[30] + self.msp[31] * self.mass0 ** 5 + self.mass0 ** self.msp[32])
        return ltms

    # 估算光度 alpha 系数
    # [已校验] Hurley_2000: equation 5.1.1(19)
    def lalphaf(self):
        mcut = 2.0
        if self.mass0 < 0.5:
            lalpha = self.msp[39]
        elif self.mass0 < 0.7:
            lalpha = self.msp[39] + ((0.3 - self.msp[39]) / 0.2) * (self.mass0 - 0.5)
        elif self.mass0 < self.msp[37]:
            lalpha = 0.3 + ((self.msp[40] - 0.3) / (self.msp[37] - 0.7)) * (self.mass0 - 0.7)
        elif self.mass0 < self.msp[38]:
            lalpha = self.msp[40] + ((self.msp[41] - self.msp[40]) / (self.msp[38] - self.msp[37])) * (
                        self.mass0 - self.msp[37])
        elif self.mass0 < mcut:
            lalpha = self.msp[41] + ((self.msp[42] - self.msp[41]) / (mcut - self.msp[38])) * (
                        self.mass0 - self.msp[38])
        else:
            lalpha = (self.msp[33] + self.msp[34] * self.mass0 ** self.msp[36]) / (
                        self.mass0 ** 0.4 + self.msp[35] * self.mass0 ** 1.9)
        return lalpha

    # 估算光度 beta 系数
    # [已校验] Hurley_2000: equation 5.1.1(20)
    def lbetaf(self):
        lbeta = max(0, self.msp[43] - self.msp[44] * self.mass0 ** self.msp[45])
        if self.mass0 > self.msp[46] and lbeta > 0:
            B = self.msp[43] - self.msp[44] * self.msp[46] ** self.msp[45]
            lbeta = max(0, B - 10 * B * (self.mass0 - self.msp[46]))
        return lbeta

    # 估算光度 neta 系数
    # [已校验] Hurley_2000: equation 5.1.1(18)
    def lnetaf(self):
        if self.mass0 <= 1:
            lneta = 10
        elif self.mass0 >= 1.1:
            lneta = 20
        else:
            lneta = 10 + 100 * (self.mass0 - 1)
        lneta = np.minimum(lneta, self.msp[97])
        return lneta

    # A function to evaluate the radius at the end of the MS
    # Note that a safety check is added to ensure Rtms > Rzams when extrapolating the function to low masses. (JH 24/11/97)
    # [已校验] Hurley_2000: equation 5.1(9)
    def rtmsf(self, m=0):
        mass = self.mass0 if m == 0 else m

        if mass <= self.msp[62]:
            rtms = (self.msp[52] + self.msp[53] * mass ** self.msp[55]) / (self.msp[54] + mass ** self.msp[56])
            # extrapolated to low mass(M < 0.5)
            rtms = max(rtms, 1.5 * self.rzamsf(mass))
        elif mass >= self.msp[62] + 0.1:
            rtms = (self.msp[57] * mass ** 3 + self.msp[58] * mass ** self.msp[61] + self.msp[59] * mass ** (
                        self.msp[61] + 1.5)) / (
                           self.msp[60] + mass ** 5)
        else:
            rtms = self.msp[63] + ((mass - self.msp[62]) / 0.1) * (self.msp[64] - self.msp[63])

        return rtms

    # 估算半径 alpha 系数
    # [已校验] Hurley_2000: equation 5.1.1(21)
    def ralphaf(self):
        if self.mass0 <= 0.5:
            ralpha = self.msp[73]
        elif self.mass0 <= 0.65:
            ralpha = self.msp[73] + ((self.msp[74] - self.msp[73]) / 0.15) * (self.mass0 - 0.5)
        elif self.mass0 <= self.msp[70]:
            ralpha = self.msp[74] + ((self.msp[75] - self.msp[74]) / (self.msp[70] - 0.65)) * (self.mass0 - 0.65)
        elif self.mass0 <= self.msp[71]:
            ralpha = self.msp[75] + ((self.msp[76] - self.msp[75]) / (self.msp[71] - self.msp[70])) * (
                        self.mass0 - self.msp[70])
        elif self.mass0 <= self.msp[72]:
            ralpha = (self.msp[65] * self.mass0 ** self.msp[67]) / (self.msp[66] + self.mass0 ** self.msp[68])
        else:
            a5 = (self.msp[65] * self.msp[72] ** self.msp[67]) / (self.msp[66] + self.msp[72] ** self.msp[68])
            ralpha = a5 + self.msp[69] * (self.mass0 - self.msp[72])
        return ralpha

    # 估算半径 beta 系数
    # [已校验] Hurley_2000: equation 5.1.1(22)
    def rbetaf(self):
        m2 = 2
        m3 = 16
        if self.mass0 <= 1:
            rbeta = 1.06
        elif self.mass0 <= self.msp[82]:
            rbeta = 1.06 + ((self.msp[81] - 1.06) / (self.msp[82] - 1)) * (self.mass0 - 1)
        elif self.mass0 <= m2:
            b2 = (self.msp[77] * m2 ** (7 / 2)) / (self.msp[78] + m2 ** self.msp[79])
            rbeta = self.msp[81] + ((b2 - self.msp[81]) / (m2 - self.msp[82])) * (self.mass0 - self.msp[82])
        elif self.mass0 <= m3:
            rbeta = (self.msp[77] * self.mass0 ** (7 / 2)) / (self.msp[78] + self.mass0 ** self.msp[79])
        else:
            b3 = (self.msp[77] * m3 ** (7 / 2)) / (self.msp[78] + m3 ** self.msp[79])
            rbeta = b3 + self.msp[80] * (self.mass0 - m3)
        rbeta = rbeta - 1
        return rbeta

    # 估算半径 gamma 系数
    # [已校验] Hurley_2000: equation 5.1.1(23)
    def rgammaf(self):
        m1 = 1
        b1 = np.maximum(0, self.msp[83] + self.msp[84] * (m1 - self.msp[85]) ** self.msp[86])
        if self.mass0 <= m1:
            rgamma = self.msp[83] + self.msp[84] * abs(self.mass0 - self.msp[85]) ** self.msp[86]
        elif m1 < self.mass0 <= self.msp[88]:
            rgamma = b1 + (self.msp[89] - b1) * ((self.mass0 - m1) / (self.msp[88] - m1)) ** self.msp[87]
        elif self.msp[88] < self.mass0 <= self.msp[88] + 0.1:
            if self.msp[88] > m1:
                b1 = self.msp[89]
            rgamma = b1 - 10 * b1 * (self.mass0 - self.msp[88])
        else:
            rgamma = 0
        rgamma = max(rgamma, 0)
        return rgamma

    # A function to evaluate the luminosity at the base of Giant Branch (for those models that have one)
    # Note that this function is only valid for LM & IM stars
    # [已校验] Hurley_2000: equation 5.1(10)
    def l_bgb(self):
        l_bgb = (self.gbp[1] * self.mass0 ** self.gbp[5] + self.gbp[2] * self.mass0 ** self.gbp[8]) / (
                 self.gbp[3] + self.gbp[4] * self.mass0 ** self.gbp[7] + self.mass0 ** self.gbp[6])
        return l_bgb

    # A function to evaluate the derivitive of the Lbgb function.
    # Note that this function is only valid for LM & IM stars
    def l_bgb_derivative(self):
        f = self.gbp[1] * self.mass0 ** self.gbp[5] + self.gbp[2] * self.mass0 ** self.gbp[8]
        df = self.gbp[5] * self.gbp[1] * self.mass0 ** (self.gbp[5] - 1) + self.gbp[8] * self.gbp[2] * self.mass0 ** (
                    self.gbp[8] - 1)
        g = self.gbp[3] + self.gbp[4] * self.mass0 ** self.gbp[7] + self.mass0 ** self.gbp[6]
        dg = self.gbp[7] * self.gbp[4] * self.mass0 ** (self.gbp[7] - 1) + self.gbp[6] * self.mass0 ** (self.gbp[6] - 1)
        l_bgb_d = (df * g - f * dg) / (g * g)
        return l_bgb_d

    # 估算 He星零龄主序的光度
    # [已校验] Hurley_2000: equation 6.1(77)
    def lzhef(self, m=0.):
        mass = self.mass0 if m == 0 else m
        lzhe = 15262 * mass ** 10.25 / (mass ** 9 + 29.54 * mass ** 7.5 + 31.18 * mass ** 6 + 0.0469)
        return lzhe

    # A function to evaluate the ZAHB luminosity for LM stars. (OP 28/01/98)
    # Continuity with LHe, min for IM stars is ensured by setting lx = lHeif(mhefl,z,0.0,1.0)*lHef(mhefl,z,mfgb)
    # and the call to lzhef ensures continuity between the ZAHB and the NHe-ZAMS as Menv -> 0.
    # [已校验] Hurley_2000: equation 5.3(53)
    def lzahbf(self, m, mc, mhefl):
        a5 = self.lzhef(mc)
        a4 = (self.gbp[69] + a5 - self.gbp[74]) / ((self.gbp[74] - a5) * np.exp(self.gbp[71] * mhefl))
        mm = max((m - mc) / (mhefl - mc), 1e-12)
        lzahb = a5 + (1 + self.gbp[72]) * self.gbp[69] * mm ** self.gbp[70] / (
                (1 + self.gbp[72] * mm ** self.gbp[73]) * (1 + a4 * np.exp(m * self.gbp[71])))
        return lzahb

    # A function to evalute the luminosity pertubation on the MS phase for M > Mhook. (JH 24/11/97)【我对这个函数的定义有改动】
    # [已校验] Hurley_2000: equation 5.1.1(16)
    def lpertf(self):
        if self.mass0 <= self.zpars[1]:
            lhook = 0
        elif self.mass0 >= self.msp[51]:
            lhook = np.minimum(self.msp[47] / self.mass0 ** self.msp[48], self.msp[49] / self.mass0 ** self.msp[50])
        else:
            B = np.minimum(self.msp[47] / self.msp[51] ** self.msp[48], self.msp[49] / self.msp[51] ** self.msp[50])
            lhook = B * ((self.mass0 - self.zpars[1]) / (self.msp[51] - self.zpars[1])) ** 0.4
        return lhook

    # A function to evalute the radius pertubation on the MS phase for M > Mhook. (JH 24/11/97)【我对这个函数的定义有改动】
    # [已校验] Hurley_2000: equation 5.1.1(17)
    def rpertf(self):
        if self.mass0 <= self.zpars[1]:
            rhook = 0
        elif self.mass0 <= self.msp[94]:
            rhook = self.msp[95] * np.sqrt((self.mass0 - self.zpars[1]) / (self.msp[94] - self.zpars[1]))
        elif self.mass0 <= 2:
            m1 = 2
            B = (self.msp[90] + self.msp[91] * m1 ** (7 / 2)) / (self.msp[92] * m1 ** 3 + m1 ** self.msp[93]) - 1
            rhook = self.msp[95] + (B - self.msp[95]) * ((self.mass0 - self.msp[94]) / (m1 - self.msp[94])) ** self.msp[
                96]
        else:
            rhook = (self.msp[90] + self.msp[91] * self.mass0 ** (7 / 2)) / (
                        self.msp[92] * self.mass0 ** 3 + self.mass0 ** self.msp[93]) - 1
        return rhook

    # A function to evaluate the BAGB luminosity. (OP 21/04/98)
    # Continuity between LM and IM functions is ensured by setting gbp(16) = lbagbf(mhefl,0.0) with gbp(16) = 1.0.
    # [已校验] Hurley_2000: equation 5.3(56) 第三行有出入
    def lbagbf(self, m=0):
        a4 = (self.gbp[9] * self.zpars[2] ** self.gbp[10] - self.gbp[16]) / (
                np.exp(self.zpars[2] * self.gbp[11]) * self.gbp[16])
        if self.mass0 < self.zpars[2]:
            lbagb = self.gbp[9] * self.mass0 ** self.gbp[10] / (1 + a4 * np.exp(self.mass0 * self.gbp[11]))
        else:
            lbagb = (self.gbp[12] + self.gbp[13] * self.mass0 ** (self.gbp[15] + 1.8)) / (
                    self.gbp[14] + self.mass0 ** self.gbp[15])
        if m > 0:
            lbagb = (self.gbp[12] + self.gbp[13] * m ** (self.gbp[15] + 1.8)) / (self.gbp[14] + m ** self.gbp[15])
        return lbagb

    # A function to evaluate He-ignition luminosity  (OP 24/11/97)
    # Continuity between the LM and IM functions is ensured with a first call setting lhefl = lHeIf(mhefl,0.0)
    # [已校验] Hurley_2000: equation 5.3(49) 第二行有出入
    def lHeIf(self, m=0):
        mass = self.mass0 if m == 0 else m
        if mass < self.zpars[2]:
            lHeI = self.gbp[38] * mass ** self.gbp[39] / (1 + self.gbp[41] * np.exp(mass * self.gbp[40]))
        else:
            lHeI = (self.gbp[42] + self.gbp[43] * mass ** 3.8) / (self.gbp[44] + mass ** 2)
        return lHeI

    # A function to evaluate the ratio LHe,min/LHeI  (OP 20/11/97)
    # Note that this function is everywhere <= 1, and is only valid for IM stars
    # [已校验] Hurley_2000: equation 5.3(51)\
    def lHef(self, m=0):
        mass = self.mass0 if m == 0 else m
        lHe = (self.gbp[45] + self.gbp[46] * mass ** (self.gbp[48] + 0.1)) / (self.gbp[47] + mass ** self.gbp[48])
        return lHe

    # 通过 Mc 估算 GB, AGB and Naked He stars 的光度
    # [已校验] Hurley_2000: equation 5.2(37)
    def mc_to_lum_gb(self, mc, GB):
        if mc <= GB[7]:
            lum = GB[4] * (mc ** GB[5])
        else:
            lum = GB[3] * (mc ** GB[6])
        return lum

    # A function to evaluate the He-burning lifetime.
    # For IM & HM stars, tHef is relative to tBGB.
    # Continuity between LM and IM stars is ensured by setting thefl = tHef(mhefl,0.0,0.0)
    # the call to themsf ensures continuity between HB and NHe stars as Menv -> 0.
    # [已校验] Hurley_2000: equation 5.3(57)
    def tHef(self, m, mc, mhefl):
        if m <= mhefl:
            mm = max((mhefl - m) / (mhefl - mc), 1e-12)
            tHe = (self.gbp[54] + (self.themsf(mc) - self.gbp[54]) * mm ** self.gbp[55]) * (
                    1 + self.gbp[57] * np.exp(m * self.gbp[56]))
        else:
            tHe = (self.gbp[58] * m ** self.gbp[61] + self.gbp[59] * m ** 5) / (self.gbp[60] + m ** 5)
        return tHe

    # 估算 He 星的主序时间
    # [已校验] Hurley_2000: equation 6.1(79)
    def themsf(self, m=0):
        if m == 0:
            thems = (0.4129 + 18.81 * self.mass0 ** 4 + 1.853 * self.mass0 ** 6) / self.mass0 ** 6.5
        else:
            thems = (0.4129 + 18.81 * m ** 4 + 1.853 * m ** 6) / m ** 6.5
        return thems

    # 通过光度估算 GB, AGB and NHe stars 的 Mc
    # [已校验] Hurley_2000: equation 5.2(37)等效
    def lum_to_mc_gb(self, lum):
        if lum <= self.lums[6]:
            mc = (lum / self.GB[4]) ** (1 / self.GB[5])
        else:
            mc = (lum / self.GB[3]) ** (1 / self.GB[6])
        return mc

    # 估算渐近巨星分支上的半径
    # [已校验] Hurley_2000: equation 5.4(74)
    def ragbf(self, m, lum, mhef):
        m1 = mhef - 0.2
        if m <= m1:
            b50 = self.gbp[19]
            A = self.gbp[29] + self.gbp[30] * m
        elif m >= mhef:
            b50 = self.gbp[19] * self.gbp[24]
            A = min(self.gbp[25] / m ** self.gbp[26], self.gbp[27] / m ** self.gbp[28])
        else:
            b50 = self.gbp[19] * (1 + (self.gbp[24] - 1) * (m - m1) / 0.2)
            A = self.gbp[31] + (self.gbp[32] - self.gbp[31]) * (m - m1) / 0.2
        ragb = A * (lum ** self.gbp[18] + self.gbp[17] * lum ** b50)
        return ragb

    # A function to evaluate core mass at BGB or He ignition for IM & HM stars
    # [已校验] Hurley_2000: equation 5.2(44)
    def mc_bgb(self, m, stage='bgb'):
        if stage == 'bgb':
            c = self.zpars[9] ** 4 - self.gbp[33] * self.zpars[2] ** self.gbp[34]
        elif stage == 'HeI':
            c = self.zpars[10] ** 4 - self.gbp[33] * self.zpars[2] ** self.gbp[34]
        else:
            raise ValueError("Unsupported evolution stage. Expected one of: 'bgb', 'HeI'.")
        mc_bagb = self.mc_bagb(m)
        mc_bgb = min(0.95 * mc_bagb, (c + self.gbp[33] * m ** self.gbp[34]) ** (1 / 4))
        return mc_bgb

    # A function to evaluate core mass at the BAGB (OP 25/11/97)
    # [已校验] Hurley_2000: equation 5.3(66)
    def mc_bagb(self, m):
        mc_bagb = (self.gbp[37] + self.gbp[35] * m ** self.gbp[36]) ** (1 / 4)
        return mc_bagb

    # A function to evaluate the initial mass given the core mass at the BGB or He ignition for IM & HM stars
    # by inverting mc_bgb.
    def mc_bgb_invert(self, mc, stage='bgb'):
        if stage == 'bgb':
            c = self.zpars[9] ** 4 - self.gbp[33] * self.zpars[2] ** self.gbp[34]
        elif stage == 'HeI':
            c = self.zpars[10] ** 4 - self.gbp[33] * self.zpars[2] ** self.gbp[34]
        else:
            raise ValueError("Unsupported evolution stage. Expected one of: 'bgb', 'HeI'.")
        m1 = self.mc_bagb_invert(mc / 0.95)
        m2 = ((mc ** 4 - c) / self.gbp[33]) ** (1 / self.gbp[34])
        m0 = max(m1, m2)
        return m0

    # A function to evaluate the initial mass given the core mass at the BAGB by inverting mc_bagb.
    def mc_bagb_invert(self, mc):
        mc4 = mc ** 4
        if mc4 > self.gbp[37]:
            m0 = ((mc4 - self.gbp[37]) / self.gbp[35]) ** (1 / self.gbp[36])
        else:
            m0 = 0
        return m0

    # A function to evaluate Mc given t for GB, AGB and NHe stars
    # [已校验] Hurley_2000: equation 5.2(34、39)
    def mcgbtf(self, t, A, GB, tinf1, tinf2, tx):
        if t <= tx:
            mcgbt = ((GB[5] - 1) * A * GB[4] * (tinf1 - t)) ** (1 / (1 - GB[5]))
        else:
            mcgbt = ((GB[6] - 1) * A * GB[3] * (tinf2 - t)) ** (1 / (1 - GB[6]))
        return mcgbt

    # A function to evaluate the minimum radius during blue loop(He-burning) for IM & HM stars
    # [已校验] Hurley_2000: equation 5.3(55)
    def rminf(self, m):
        rmin = (self.gbp[49] * m + (self.gbp[50] * m) ** self.gbp[52] * m ** self.gbp[53]) / (
                    self.gbp[51] + m ** self.gbp[53])
        return rmin

    # 估算巨星分支上的半径
    # [已校验] Hurley_2000: equation 5.2(46)
    def rgbf(self, m, lum):
        a = min(self.gbp[20] / m ** self.gbp[21], self.gbp[22] / m ** self.gbp[23])
        rgb = a * (lum ** self.gbp[18] + self.gbp[17] * lum ** self.gbp[19])
        return rgb

    # 估算低质量恒星的零龄水平分支(ZAHB)半径
    # Continuity with R(LHe,min) for IM stars is ensured by setting lx = lHeif(mhefl,z,0.0,1.0)*lHef(mhefl,z,mfgb),
    # and the call to rzhef ensures continuity between the ZAHB and the NHe-ZAMS as Menv -> 0.
    # [已校验] Hurley_2000: equation 5.3(54)
    def rzahbf(self, m, mc, mhefl):
        rx = self.rzhef(mc)
        ry = self.rgbf(m, self.lzahbf(m, mc, mhefl))
        mm = max((m - mc) / (mhefl - mc), 1e-12)
        f = (1 + self.gbp[76]) * mm ** self.gbp[75] / (1 + self.gbp[76] * mm ** self.gbp[77])
        rzahb = (1 - f) * rx + f * ry
        return rzahb

    # 估算 He 星零龄主序的半径
    # [已校验] Hurley_2000: equation 6.1(78)
    def rzhef(self, m):
        rzhe = 0.2391 * m ** 4.6 / (m ** 4 + 0.162 * m ** 3 + 0.0065)
        return rzhe

    # A function to evaluate radius derivitive on the GB (as f(L)).  [全局无调用]
    def rgbdf(self, m, lum, x):
        a1 = min(x.gbp[20] / m ** x.gbp[21], x.gbp[22] / m ** x.gbp[23])
        rgbd = a1 * (x.gbp[18] * lum ** (x.gbp[18] - 1) + x.gbp[17] * x.gbp[19] * lum ** (x.gbp[19] - 1))
        return rgbd

    # A function to evaluate radius derivitive on the AGB (as f(L)). [全局无调用]
    def ragbdf(self, m, lum, mhelf, x):
        m1 = mhelf - 0.2
        if m >= mhelf:
            xx = x.gbp[24]
        elif m >= m1:
            xx = 1 + 5 * (x.gbp[24] - 1) * (m - m1)
        else:
            xx = 1
        a4 = xx * x.gbp[19]
        if m <= m1:
            a1 = x.gbp[29] + x.gbp[30] * m
        elif m >= mhelf:
            a1 = min(x.gbp[25] / m ** x.gbp[26], x.gbp[27] / m ** x.gbp[28])
        else:
            a1 = x.gbp[31] + 5 * (x.gbp[32] - x.gbp[31]) * (m - m1)
        ragbd = a1 * (x.gbp[18] * lum ** (x.gbp[18] - 1) + x.gbp[17] * a4 * lum ** (a4 - 1))
        return ragbd

    # A function to evaluate core mass at the end of the MS as a fraction of the BGB value,
    # i.e. this must be multiplied by the BGB value (see below) to give the actual core mass.
    # [已校验] Hurley_2000: equation 5.1.2(29)
    def mctmsf(self):
        mctms = (1.586 + self.mass0 ** 5.25) / (2.434 + 1.02 * self.mass0 ** 5.25)
        return mctms

    # A function to evaluate L given t for GB, AGB and NHe stars
    # [已校验] Hurley_2000: equation 5.2(35)
    def lgbtf(self, A):
        if self.age <= self.tscls[6]:
            lgbt = self.GB[4] * (((self.GB[5] - 1) * A * self.GB[4] * (self.tscls[4] - self.age)) ** (
                        self.GB[5] / (1 - self.GB[5])))
        else:
            lgbt = self.GB[3] * (((self.GB[6] - 1) * A * self.GB[3] * (self.tscls[5] - self.age)) ** (
                    self.GB[6] / (1 - self.GB[6])))
        return lgbt

    # A function to evaluate the blue-loop fraction of the He-burning lifetime for IM & HM stars  (OP 28/01/98)
    # [已校验] Hurley_2000: equation 5.3(58) 有些不太一样
    def tblf(self):
        mr = self.zpars[2] / self.zpars[3]
        if self.mass0 <= self.zpars[3]:
            m1 = self.mass0 / self.zpars[3]
            m2 = np.log10(m1) / np.log10(mr)
            m2 = max(m2, 1e-12)
            tbl = self.gbp[64] * m1 ** self.gbp[63] + self.gbp[65] * m2 ** self.gbp[62]
        else:
            r1 = 1 - self.rminf(self.mass0) / self.ragbf(self.mass0, self.lHeIf(), self.zpars[2])
            r1 = max(r1, 1e-12)
            tbl = self.gbp[66] * self.mass0 ** self.gbp[67] * r1 ** self.gbp[68]
        tbl = min(1, max(0, tbl))
        if tbl < 1e-10:
            tbl = 0
        return tbl

    # 估算 He 星主序上的光度
    # [已校验] Hurley_2000: equation 6.1(78) [无调用]
    def l_He_MS(self, m):
        lzhe = 15262 * m ** 10.25 / (m ** 9 + 29.54 * m ** 7.5 + 31.18 * m ** 6 + 0.0469)
        return lzhe

    # 根据质量、光度估算赫氏空隙中 He 星的半径
    def rhehgf(self, m, lum, rzhe, lthe):
        Lambda = 500 * (2 + m ** 5) / m ** 2.5
        rhehg = rzhe * (lum / lthe) ** 0.2 + 0.02 * (np.exp(lum / Lambda) - np.exp(lthe / Lambda))
        return rhehg

    # 估算 He 巨星的半径
    def rhegbf(self, lum):
        rhegb = 0.08 * lum ** (3 / 4)
        return rhegb


    # 根据BGB时的核质量, 求解初始质量
    def solve_initial_mass_GB(self, mc, max_iterations=100, tolerance=1e-4):
        # 当核质量超过最大允许的BGB核质量(初始质量为M_FGB), 改变类型为CHeB恒星
        mc_bgb_m_fgb = self.mc_bgb(self.zpars[3])
        if mc >= mc_bgb_m_fgb:
            self.type = 4
            self.solve_initial_mass_CHeB(mc, 0)
            return

        # 在BGB处的最大简并核质量
        mc_bgb_m_hef = self.mc_bgb(self.zpars[2])
        if mc >= mc_bgb_m_hef:
            self.mass0 = self.mc_bgb_invert(mc)
        else:
            self.mass0 = self.zpars[2]
            self.StellarCal()
            lum = self.mc_to_lum_gb(mc, self.GB)
            for _ in range(max_iterations):
                delta_l = self.l_bgb() - lum
                if abs(delta_l / lum) <= tolerance:
                    break
                l_bgb_d = self.l_bgb_derivative()
                self.mass0 = self.mass0 - delta_l / l_bgb_d

    # 根据CHeB时的氦核质量, 求解初始质量
    def solve_initial_mass_CHeB(self, mc, age_frac):
        # 最小初始质量, 假设此时的核质量是刚到达BAGB时核质量
        mc_bagb_m_hef = self.mc_bagb(self.zpars[2])
        if mc >= mc_bagb_m_hef:
            m_min = self.mc_bagb_invert(mc)
        else:
            m_min = self.zpars[2]

        # 最大初始质量, 假设此时氦核刚刚点燃
        m_max = self.mc_bgb_invert(mc, stage='HeI')

        # 迭代计算初始质量 (参考Hurley et.al 2002 equation (84))
        fmid = (1.0 - age_frac) * self.mc_bgb(m_max, stage='HeI') + age_frac * self.mc_bagb(m_max) - mc
        f = (1.0 - age_frac) * self.mc_bgb(m_min, stage='HeI') + age_frac * self.mc_bagb(m_min) - mc
        if f * fmid >= 0.0:
            self.type = 3
            self.mass0 = m_min
            return

        m0 = m_min
        dm = m_max - m_min
        for j in range(100):
            dm = 0.50 * dm
            mmid = m0 + dm
            fmid = (1.0 - age_frac) * self.mc_bgb(mmid, stage='HeI') + age_frac * self.mc_bagb(mmid) - mc
            if fmid < 0.0:
                m0 = mmid
            if abs(dm) < 0.00001 or abs(fmid) < tiny:
                break
        self.mass0 = m0


    # 根据BAGB时的氦核质量, 求解初始质量
    def solve_initial_mass_EAGB(self, mc):
        m0 = self.mc_bagb_invert(mc)
        if m0 <= 0:
            # 根据eq.(66) of Hurley et al. 2000, BAGB时的最小氦核质量是0.5114M_sun,
            # 如果合并后的氦核质量比这个值还小, 简单假设初始质量为1 M_sun
            self.mass0 = 1.0
        else:
            self.mass0 = m0

    # 根据TPAGB时的He核质量, 求解初始质量
    def solve_initial_mass_TPAGB(self, mc):
        mc_du = 0.44 * 2.25 + 0.448
        if mc > mc_du:
            mc_He = (mc + 0.35) / 0.773
        elif mc >= 0.8:
            mc_He = (mc - 0.448) / 0.44
        else:
            mc_He = mc

        m0 = self.mc_bagb_invert(mc_He)
        if m0 <= 0:
            # 核质量太小, 无法求解, 简单假设初始质量为1 M_sun
            self.mass0 = 1
        else:
            self.mass0 = m0

    # 根据HeHG/HeGB时的CO核质量, 求解初始质量
    def solve_initial_mass_HeGB(self, mc, max_iterations=100, tolerance=1e-2):
        # 将新的恒星放在主序末的位置, 用迭代法 (初始值 → 主序末光度 → 核质量 → 比较实际值调整初始质量)
        # 假设初始质量 = 当前质量
        m0 = self.mass
        for _ in range(max_iterations):
            self.mass0 = m0
            self.StellarCal()
            mc_current = self.lum_to_mc_gb(self.lums[2])
            if abs(mc_current - mc) < tolerance:
                return
            else:
                ratio = mc / mc_current
                ratio = max(0.5, min(2.0, ratio))
                m0 = m0 * ratio




