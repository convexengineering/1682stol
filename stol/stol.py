import os
import pandas as pd
from gpkit import Model, parse_variables, Vectorize, SignomialEquality
from gpkit.constraints.tight import Tight as TCS
from gpfit.fit_constraintset import FitCS
import gpkit
import math
import numpy as np
import matplotlib.pyplot as plt
# from gpkitmodels.GP.aircraft.wing.wing import Wing
from gpkitmodels import g
from gpkitmodels.GP.aircraft.wing.capspar import CapSpar
from gpkitmodels.GP.aircraft.wing.wing_skin import WingSkin
from gpkitmodels.GP.aircraft.wing.wing import Planform
from gpkitmodels.GP.aircraft.tail.horizontal_tail import HorizontalTail
from gpkitmodels.GP.aircraft.tail.vertical_tail import VerticalTail
from gpkitmodels.GP.aircraft.tail.tail_boom import *

from decimal import *
from sens_chart import *
pi = math.pi

class Aircraft(Model):
    """ Aircraft

    Variables
    ---------
    mass                [kg]    aircraft mass
    n_pax       4       [-]     number of passengers
    mpax        93      [kg]    single passenger mass
    mbaggage    9       [kg]    single baggage mass
    tangamma    0.5     [-]     tan of the gamma climb angle
    d                   [in]    spar diam
    """
    def setup(self,poweredwheels=False,n_wheels=3,hybrid=False):
        exec parse_variables(Aircraft.__doc__)

        self.equipment = Equipment()
        self.battery = Battery()
        self.fuselage = Fuselage()
        self.bw = BlownWing()
        self.cabin = Cabin()
        self.htail = HorizontalTail()
        self.vtail = VerticalTail()
        self.boom =  TailBoom()
        self.gear = Gear()
        self.vtail.substitutions[self.vtail.planform.tau] = 0.08
        self.htail.substitutions[self.htail.planform.tau] = 0.08
        self.htail.substitutions[self.htail.mh] = 0.8
        self.htail.substitutions[self.vtail.Vv] = 0.1
        self.htail.substitutions[self.htail.planform.CLmax] = 3
        self.vtail.substitutions[self.vtail.planform.CLmax] = 3

        self.components = [self.cabin,self.bw,self.battery,self.fuselage,self.gear,self.equipment]
    
        if hybrid:
            self.tank = Tank()
            self.genandic = GenAndIC()
            self.components += [self.tank,self.genandic]

        if poweredwheels:
                self.pw = PoweredWheel()
                self.wheels = n_wheels*[self.pw]
                self.components += self.wheels

        constraints = [

                       self.htail.Vh == (self.htail["S"]*self.htail.lh/self.bw.wing["S"]**2 *self.bw.wing["b"]),
                       self.vtail.Vv == (self.vtail["S"]*self.vtail.lv/self.bw.wing["S"]/self.bw.wing["b"]),
                       # self.vtail.planform["b"] >= Variable("bv",48,"in"),
                       # self.vtail.planform["croot"] >= Variable("croot",27,"in"),

                       # self.htail.planform["b"] >= Variable("bh",60,"in"),
                       # self.htail.planform["croot"] >= Variable("croot",27,"in"),
                       self.boom["l"] >= self.htail.lh + self.htail.planform.croot,
                       self.boom["l"] >= self.vtail.lv + self.vtail.planform.croot,

                       # self.boom["l"] <= Variable("a",20,"ft"),
                       # self.vtail.lv == Variable("lv",180,"in"),
                       # self.htail.lh == Variable("lh",180,"in"),

                       self.fuselage.m >= 0.17*(sum(c.topvar("m") for c in self.components) + (self.vtail.W + self.htail.W)/g + (mpax+mbaggage)*n_pax),
                       self.mass>=sum(c.topvar("m") for c in self.components) + (self.boom.W + self.vtail.W + self.htail.W)/g + (mpax+mbaggage)*n_pax]

        for s in self.boom.d:
            constraints+=[s == d]
        with gpkit.SignomialsEnabled():
            constraints += [self.boom["l"]*0.36 <= self.gear.l + self.fuselage.h,
                            self.bw.wing.b - Variable("w_fuse",50,"in") >= self.bw.n_prop*2*self.bw.powertrain.r]
        return constraints, self.components, self.htail, self.vtail, self.boom

    def dynamic(self,state,hybrid=False,powermode="batt-chrg",t_charge=None,groundroll=False):
        return AircraftP(self,state,hybrid,powermode=powermode,t_charge=t_charge,groundroll=groundroll)
    def loading(self,state,Wcent):
        return AircraftLoading(self,state)

class AircraftP(Model):
    """ AircraftP

    Variables
    ---------
    P                   [kW]    total power draw
    CD                  [-]     total CD, referenced to wing area
    P_charge            [kW]    battery charging power
    P_avionics  0.25    [kW]    avionics continuous power draw
    C_chrg              [1/hr]  battery charge rate
    CDfrac              [-]     fuselage drag fraction
    """
    def setup(self,aircraft,state,hybrid=False,powermode="batt-chrg",t_charge=None,groundroll=False):
        exec parse_variables(AircraftP.__doc__)
        self.fuse_perf = aircraft.fuselage.dynamic(state)

        self.bw_perf = aircraft.bw.dynamic(state)
        self.batt_perf = aircraft.battery.dynamic(state)
        self.htail_perf = aircraft.htail.flight_model(aircraft.htail, state)
        self.vtail_perf = aircraft.vtail.flight_model(aircraft.vtail, state)
        self.boom_perf = aircraft.boom.flight_model(aircraft.boom, state)
        self.perf_models = [self.fuse_perf,self.bw_perf,self.batt_perf,self.htail_perf,self.vtail_perf,self.boom_perf]
        self.fs = state

        constraints = [P >= self.bw_perf["P"] + P_avionics,
                       CD >= self.bw_perf.C_D + ((aircraft.htail.planform.S/aircraft.bw.wing.planform.S)*self.htail_perf.Cd + (self.fuse_perf.Cd*aircraft.fuselage.Swet/aircraft.bw.wing.planform.S) + (aircraft.vtail.planform.S/aircraft.bw.wing.planform.S)*self.vtail_perf.Cd) + (aircraft.boom.S/aircraft.bw.wing.planform.S)*self.boom_perf.Cf,
                       self.bw_perf.C_T >= CD,
                       CDfrac == (self.fuse_perf.Cd*aircraft.fuselage.Swet/aircraft.bw.wing.planform.S)/CD
                    ]

        #If we're not in groundroll, apply lift=weight and fuselage drag
        if groundroll == False:
            constraints += [0.5*self.bw_perf.C_L*state.rho*aircraft.bw.wing["S"]*state.V**2 >= aircraft.mass*g]

        if hybrid:
            self.gen_perf = aircraft.genandic.dynamic(state)
            if powermode == "batt-chrg":
                constraints += [#C_chrg == P_charge/E_capacity,
                                P_charge >= aircraft.battery.E_capacity/t_charge,
                                self.gen_perf.P_out >= P + P_charge,
                                self.batt_perf.P >= Variable("P_draw_batt",1e-4,"W")]
            if powermode == "batt-dischrg":
                with gpkit.SignomialsEnabled():
                    constraints += [self.batt_perf.P + self.gen_perf.P_out >= P]
            self.perf_models += [self.gen_perf]
        else:
            constraints += [self.batt_perf.P >= P]

        return constraints,self.perf_models

class AircraftLoading(Model):
    def setup(self,aircraft,state):
        hbend = aircraft.boom.tailLoad(aircraft.boom,aircraft.htail,state)
        vbend = aircraft.boom.tailLoad(aircraft.boom,aircraft.vtail,state)
        self.wingl = aircraft.bw.wing.spar.loading(aircraft.bw.wing, state)
        loading = [self.wingl,hbend,vbend]
        return loading

class Gear(Model):
    """Gear
    Variables
    ---------
    m    78.08 [lb]    total landing gear mass
    l    3     [ft]    landing gear length
    """
    def setup(self):
        #values are from C172 POH
        exec parse_variables(Gear.__doc__)
        return []

class Cabin(Model):
    """Cabin
    Variables
    ---------
    m        78.43     [kg]       total cabin mass
    """
    def setup(self):
        exec parse_variables(Cabin.__doc__)
        return []

class Equipment(Model):
    """Equipment
    Variables
    ---------
    m        304.25     [lb]       total equipment mass, without battery
    """
    def setup(self):
        exec parse_variables(Equipment.__doc__)
        return []

class Fuselage(Model):
    """ Fuselage

    Variables
    ---------
    m                   [lb]    mass of fuselage
    l       12          [ft]    fuselage length
    w       6           [ft]    width
    h       6.4         [ft]    fuselage height
    f                   [-]     fineness ratio
    Swet    200.55      [ft^2]  wetted area of fuselage    
    """
    def setup(self):
        exec parse_variables(Fuselage.__doc__)
        return [#f <= l/w,
                f == l/h]
    def dynamic(self,state):
        return FuselageP(self,state)

class FuselageP(Model):
    """FuselageP
    Variables
    ---------
    Cd              [-]     drag coefficient
    FF              [-]     form factor
    C_f             [-]     friction coefficient
    mfac     1.1    [-]     friction margin
    Re              [-]     Reynolds number
    """
    def setup(self,fuse,state):
        exec parse_variables(FuselageP.__doc__)
        constraints = [#FF >= 1 + 60.0/(fuse.f)**3 + fuse.f/400.0,
                       FF == 12.0/6.4,
                       C_f >= 0.455/Re**0.3,
                       Cd/mfac == C_f*FF,
                       Re == state["V"]*state["rho"]*fuse.l/state["mu"],
                    ]
        return constraints

#Combination of electric motor and motor controller
class Powertrain(Model):
    """ Powertrain
    Variables
    ---------
    m                       [kg]            single powertrain mass
    Pmax                    [kW]            maximum power
    P_m_sp_cont             [W/kg]          continuous motor specific power
    P_m_sp_max              [W/kg]          maximum motor specific power
    tau_sp_max              [N*m/kg]        max specific torque
    RPMmax                  [rpm]           max rpm
    r                       [m]             propeller radius
    Pstar_ref     1         [W]             specific motor power reference
    m_ref         1         [kg]            reference motor power
    eta                     [-]             powertrain efficiency
    a             1         [W/kg]          dummy
    RPM_margin    0.9       [-]             margin for rpm
    tau_margin    0.95      [-]             margin for torque
    P_margin      0.4       [-]             margin for power (helpful for tail sizing)
    """

    def setup(self):
        exec parse_variables(Powertrain.__doc__)
                       
        with gpkit.SignomialsEnabled():
            constraints = [P_m_sp_cont <= P_margin*((Variable("a",61.8,"W/kg**2","a")*m + Variable("b",6290,"W/kg","a"))), #magicALL motor fits
                           P_m_sp_max <= P_margin*((Variable("c",86.2,"W/kg**2","c")*m + Variable("d",7860,"W/kg","d"))),  #magicALL motor fits
                           eta/Variable("f",1,"1/kg**0.0134","wing motor eta fit constant") <= 0.906*m**(0.0134),
                           (RPMmax/RPM_margin)*m**(0.201) == Variable("g",7939,"rpm*kg**0.201","g"),
                           Pmax <= m*P_m_sp_max]
        return constraints

class PoweredWheel(Model):
    """Powered Wheels

    Variables
    ---------
    RPMmax                  [rpm]       maximum RPM of motor
    gear_ratio              [-]         gear ratio of powered wheel
    gear_ratio_max  20      [-]         max gear ratio of powered wheel
    tau_max                 [N*m]       torque of the
    m                       [kg]        powered wheel motor mass
    m_ref           1       [kg]        reference mass for equations
    r               0.2     [m]         tire radius
    Pstar           5       [kW/kg]     specific power for powered wheel
    """
    def setup(self):
        exec parse_variables(PoweredWheel.__doc__)
        #Currently using the worst values
        with gpkit.SignomialsEnabled():
            constraints = [gear_ratio <= gear_ratio_max,
                           RPMmax*m**(0.201) == Variable("g",7145,"rpm*kg**0.201"),
                           tau_max/m + Variable("d",2.20e-3,"N*m/kg**3")*m**2 <= Variable("e",0.447,"N*m/kg**2")*m + Variable("f",6.71,"N*m/kg"),
                           # tau_max <= Variable("tau_m",1e-4,'N*m/kg')*m,
                           # Pstar*m <= tau_max*RPMmax
                           ]
        return constraints
    def dynamic(self,state):
        return PoweredWheelP(self,state)

class PoweredWheelP(Model):
    """ PoweredWheelsP
    Variables
    ---------
    RPM                     [rpm]       rpm of powered wheel
    tau                     [N*m]       torque of powered wheel
    T                       [N]         thrust from powered wheel
    P                       [kW]        power draw of powered wheel
    eta_mc          0.98    [-]         motor controller efficiency
    eta_m           0.9     [-]         motor efficiency
    """
    def setup(self,pw,state):
        exec parse_variables(PoweredWheelP.__doc__)
        constraints =[RPM <= pw.RPMmax,
                      state.V <= RPM*pw.r/pw.gear_ratio,
                      T <= tau*pw.gear_ratio/pw.r,
                      (P*eta_mc*eta_m) >= RPM*tau,
                      tau <= pw.tau_max]
        return constraints

class GenAndIC(Model):
    """ GenAndIC Model
    Variables
    ---------
    P_turb_sp_cont  160/61.3       [kW/kg]    specific cont power of IC (2.8 if turboshaft)
    m_g             49.5           [kg]       49.5 turbogen mass
    m_gc                           [kg]       turbogen controller mass
    m_turb          61.3           [kg]       piston mass
    P_g_sp_cont                    [W/kg]     turbogen spec power (cont)
    P_g_cont                       [W]        turbogen cont. power
    P_turb_cont                    [kW]       turboshaft continous power  
    m                              [kg]       total mass
    m_ref           1              [kg]       reference mass, for meeting units constraints
    Pstar_ref       1              [W/kg]     reference specific power, for meeting units constraints
    eta_turb        0.15           [-]        turboshaft efficiency
    """
    def setup(self):
        exec parse_variables(GenAndIC.__doc__)
        with gpkit.SignomialsEnabled():
            constraints = [P_g_sp_cont <= (Variable("a",46.4,"W/kg**2")*m_g + Variable("b",5032,"W/kg")), #magicALL motor fits
                           P_g_cont    <=   P_g_sp_cont*m_g,
                           P_turb_cont <= P_turb_sp_cont*m_turb,
                           m >= m_g + m_turb
            ]

        return constraints
    def dynamic(self,state):
        return genandicP(self,state)

class genandicP(Model):
    """genandicP Model
    Variables
    ---------
    P_g                             [kW]        generator power
    P_gc                            [kW]        generator controller power
    P_turb                          [kW]        turboshaft power
    P_fuel                          [kW]        power coming in from fuel flow
    P_out                           [kW]        output from generator controller after efficiency
    eta_wiring      0.98            [-]         efficiency of electrical connections (wiring loss)
    eta_shaft       0.98            [-]         shaft losses (two 99% efficient bearings)
    eta_g           0.953           [-]         generator efficiency
    """
    def setup(self,gen,state):
        exec parse_variables(genandicP.__doc__)
        constraints = [P_g <= gen.P_g_cont,
                       P_turb <= gen.P_turb_cont,
                       P_fuel*gen.eta_turb == P_turb,
                       P_turb*eta_shaft == P_g,
                       P_g*eta_g == P_out,
                       ]
        return constraints

class Tank(Model):
    """Tank Model
    Variables
    ---------
    m                          [lb]          total tank mass
    m_fuel                     [lb]          fuel mass
    E                          [Wh]          fuel energy
    Estar_fuel          11.9   [kWh/kg]      fuel specific energy
    V_fuel                     [gal]         fuel volume
    rho_fuel            6.7    [lb/gal]      fuel density
    m_tank                     [kg]          fuel tank mass
    rho_tank            0.55   [lb/gal]      empty tank mass (structure mass) per volume
    """
    def setup(self):
        exec parse_variables(Tank.__doc__)
        constraints = [V_fuel*rho_fuel == m_fuel,
                       m_tank >= 0.1*m_fuel,
                       m_fuel >= E/Estar_fuel,
                       m >= m_fuel+m_tank]
        return constraints


class Battery(Model):
    """ Battery
    Variables
    ---------
    m                   [kg]            total battery mass
    Estar       140     [Wh/kg]         specific energy
    E_capacity          [Wh]            energy capacity
    P_max_cont  3.9e3   [W/kg]          4.2e3, continuous power output
    P_max_burst 3.9e3   [W/kg]          5.6e3, burst power output
    eta_pack    0.8     [-]             add packing efficiency for battery
    """

    def setup(self):
        exec parse_variables(Battery.__doc__)
        with gpkit.SignomialsEnabled():
            constraints = [m >= E_capacity/(eta_pack*Estar),
                           # (P_max_cont/Variable("a",1,"W/kg") - 513.49)*(1+(Estar/Variable("b",40.9911,"Wh/kg"))**(11.79229)) <=  6.17e9,
                           # (P_max_burst/Variable("a",1,"W/kg") - 944.0619)*(1+(Estar/Variable("b",38.21934,"Wh/kg"))**(11.15887)) <=  1.02e10
                        ]

        return constraints
    def dynamic(self,state):
        return BatteryP(self,state)

class BatteryP(Model):
    """BatteryP
    Variables
    ---------
    P                   [kW]        battery power draw
    """
    def setup(self,batt,state):
        exec parse_variables(BatteryP.__doc__)
        constraints = [P <= batt.m*batt.P_max_burst]
        return constraints

class Wing(Model):
    """
    Wing Model
    Variables
    ---------
    W                   [lbf]       wing weight
    mfac        1.2     [-]         wing weight margin factor
    n_plies     5       [-]         number of plies on skin
    Upper Unbounded
    ---------------
    W
    Lower Unbounded
    ---------------
    b, Sy
    LaTex Strings
    -------------
    mfac                m_{\\mathrm{fac}}

    """

    sparModel = CapSpar
    fillModel = False
    skinModel = WingSkin

    def setup(self, N=4):
        exec parse_variables(Wing.__doc__)

        self.N = N

        self.planform = Planform(N)
        self.b = self.planform.b
        self.components = []

        if self.skinModel:
            self.skin = self.skinModel(self.planform)
            self.components.extend([self.skin])
        if self.sparModel:
            self.spar = self.sparModel(N, self.planform)
            self.components.extend([self.spar])
            self.Sy = self.spar.Sy
        if self.fillModel:
            self.foam = self.fillModel(self.planform)
            self.components.extend([self.foam])

        constraints = [
        self.spar.t[0:-2] == self.spar.t[-1],
        self.spar.w[0:-2] == self.spar.w[-1],
        self.skin.t >= n_plies*self.skin.material.tmin,
        W/mfac >= sum(c["W"] for c in self.components)]
        return constraints, self.planform, self.components

class BlownWing(Model):
    """
    Variables
    ---------
    n_prop     10   [-]             number of powertrains/propellers
    m               [kg]            mass
    """

    def setup(self):
        exec parse_variables(BlownWing.__doc__)
        self.powertrain = Powertrain()
        N = 14
        self.wing = Wing(N)
        self.wing.substitutions[self.wing.planform.tau]=0.12
        self.wing.substitutions[self.wing.planform.lam]=1

        constraints = [
            m >= n_prop*self.powertrain["m"] + self.wing.topvar("W")/g,
        ]
        return constraints,self.powertrain,self.wing
    def dynamic(self,state):
        return BlownWingP(self,state)

class BlownWingP(Model):
    #Built from Mark Drela's Powered-Lift and Drag Calculation
    #and Thin Airfoil Theory for Blown 2D Airfoils notes

    """
    Variables
    ---------
    C_L             [-]             total lift coefficient
    C_LC            [-]             lift coefficient due to circulation
    C_Q             [-]             mass momentum coefficient
    C_J             [-]             jet momentum coefficient
    C_E             [-]             energy momentum coefficient
    C_Di            [-]             induced drag coefficient
    C_Dp            [-]             profile drag
    C_D             [-]             total drag coefficient
    C_T             [-]             thrust coefficient
    Re              [-]             Reynolds number
    e       0.8     [-]             span efficiency
    mfac    1.1     [-]             profile drag margin factor
    m_dotprime      [kg/(m*s)]      jet mass flow per unit span
    J_prime         [kg/(s**2)]     momentum flow per unit span
    E_prime         [J/(m*s)]       energy flow per unit span
    rho_j   1.225   [kg/m**3]       density in jet flow
    u_j             [m/s]           velocity in jet flow
    h               [m]             Wake height
    T               [N]             propeller thrust
    P               [kW]            power draw
    eta_prop        [-]             prop efficiency loss after blade disk actuator
    A_disk          [m**2]          area of prop disk
    Mlim      0.5   [-]             tip limit
    a         343   [m/s]           speed of sound at sea level
    k_t       0.2   [-]             propeller torque coefficient
    RPMmax          [rpm]           maximum rpm of propeller
    Kf        1.180 [-]             form factor  
    C_f             [-]             friction coefficient        
    CLCmax    3.5   [-]             clc max  
    """
    def setup(self,bw,state):
        #bw is a BlownWing object
        #state is a FlightState
        exec parse_variables(BlownWingP.__doc__)
        with gpkit.SignomialsEnabled():
            constraints = [
            A_disk == bw.n_prop*pi*bw.powertrain.r**2,
            ((P*eta_prop*bw.powertrain.eta)/(0.5*T*state["V"]) - 1)**2 >= (T/(A_disk*(state.V**2)*state.rho/2)+1),
            (u_j/state.V)**2 <= (T/(A_disk*(state.V**2)*state.rho/2) + 1),
            u_j >= state.V,
            P <= bw.n_prop*bw.powertrain["Pmax"],

            C_L <= C_LC*(1+2*C_J/(pi*bw.wing["AR"]*e)),
            C_LC <= CLCmax,
            bw.powertrain.RPMmax*bw.powertrain.r <= a*Mlim,
            C_T == T/((0.5*state.rho*bw.wing["S"]*state.V**2)),
            m_dotprime == rho_j*u_j*h,
            J_prime ==  m_dotprime*u_j,
            E_prime == 0.5*m_dotprime*u_j**2,
            C_Q ==  m_dotprime/(state.rho*state.V* bw.wing["cmac"]),
            C_J == J_prime/(0.5*state.rho*state.V**2 * bw.wing["cmac"]),
            C_E == E_prime/(0.5*state.rho*state.V**3 * bw.wing["cmac"]),
            h == pi*bw.powertrain.r/2,
            C_Di*(pi*bw.wing["AR"]*e + 2*C_J) >= (C_L**2),
            # C_Di <= (C_LC**2)/(pi*bw.wing["AR"]*e),
            C_D >= C_Di  + C_Dp,
            C_f**5 == (mfac*0.074)**5 /(Re),
            C_Dp == C_f*2.1*Kf,
            Re == state["V"]*state["rho"]*(bw.wing["S"]/bw.wing["AR"])**0.5/state["mu"],
            # C_T >= C_D #steady level non-accelerating constraint as C_T-C_D = 1
            ]
        return constraints
class FlightState(Model):
    """ Flight State

    Variables
    ---------
    rho         1.225       [kg/m**3]        air density
    mu          1.789e-5    [N*s/m^2]        air viscosity
    V                       [knots]         speed
    qne                     [kg/s^2/m]      never exceed dynamic pressure
    Vne         175         [kts]           never exceed speed
    """
    def setup(self):
        exec parse_variables(FlightState.__doc__)
        return [qne == 0.5*rho*Vne**2]

class TakeOff(Model):
    """
    take off model
    http://www.dept.aoe.vt.edu/~lutze/AOE3104/takeoff&landing.pdf

    Variables
    ---------
    A                       [m/s**2]    log fit equation helper 1
    B                       [1/m]       log fit equation helper 2
    g           9.81        [m/s**2]    gravitational constant
    mu          0.025       [-]         coefficient of rolling resistance
    T                       [N]         take off thrust
    cda         0.015       [-]         parasite drag coefficient
    CDg                     [-]         drag ground coefficient
    CLg         2.5         [-]         lift coefficient during ground run
    cdp         0.025       [-]         profile drag at Vstallx1.2
    Kg          0.04        [-]         ground-effect induced drag parameter
    zsto                    [-]         take off distance helper variable
    Sto                     [ft]        take off distance
    W                       [N]         aircraft weight
    mu_friction 0.6         [-]         traction limit for powered wheels
    t                       [s]         time of takeoff maneuver
    dV                      [kt]        difference in velocity over run
    mstall      1.1         [-]         stall margin on takeoff
    rho                     [kg/m^3]    air density
    S                       [m^2]       wing area
    a                       [m/s/s]      acceleration of segment
    """
    def setup(self, aircraft,poweredwheels,n_wheels,hybrid=False,N=5):
        exec parse_variables(TakeOff.__doc__)

        self.fs = FlightState()

        path = os.path.dirname(os.path.abspath(__file__))
        df = pd.read_csv(path + os.sep + "logfit.csv")
        fd = df.to_dict(orient="records")[0] #fit data

        Pmax = aircraft.bw.powertrain.Pmax
        AR = aircraft.bw.wing.planform.AR
        perf = aircraft.dynamic(self.fs,hybrid,powermode="batt-dischrg",groundroll=True)
        self.perf = perf
        e = perf.bw_perf.e
        with gpkit.SignomialsEnabled():
            constraints = [
                    
                    rho == self.fs.rho,
                    S == aircraft.bw.wing["S"],
                    W == aircraft.mass*g,
                    self.perf.bw_perf.eta_prop == 0.75,
                    # T/W >= A/g + mu,
                    CDg == perf.bw_perf.C_D,
                    (T-0.5*CDg*rho*S*self.fs.V**2)/aircraft.mass >= a,
                    t*a == dV,
                    # FitCS(fd, zsto, [A/g, B*dV**2/g]), #fit constraint set, pass in fit data, zsto is the 
                    # # y variable, then arr of independent (input) vars, watch the units
                    # Sto >= 1.0/2.0/B*zsto,
                    ]
        # with gpkit.SignomialsEnabled():
        #     constraints += [B >= g/W*0.5*rho*S*(CDg)]
        if poweredwheels:
            wheel_models = [wheel.dynamic(fs) for wheel in aircraft.wheels]
            with gpkit.SignomialsEnabled():
                constraints += [T <= perf.bw_perf.T + sum(model.T for model in wheel_models),
                                perf.P >= perf.bw_perf.P + sum(model.P for model in wheel_models)
                            ]
                for model in wheel_models:
                    constraints += [model.T <= (aircraft.mass*g*mu_friction)/Variable("a",len(wheel_models),"-")]
                constraints += wheel_models

        else:
            constraints += [T <= perf.bw_perf.T]

        return constraints, self.fs, perf

class Climb(Model):

    """ Climb model

    Variables
    ---------
    Sclimb                  [ft]        distance covered in climb
    h_gain                  [ft]        height gained in climb
    t                       [s]         time of climb
    h_dot                   [m/s]       climb rate
    W                       [N]         aircraft weight
    
    LaTex Strings
    -------------
    Sclimb      S_{\\mathrm{climb}}
    h_gain      h_{\\mathrm{gain}}
    h_dot       \dot{h}
    """

    def setup(self,aircraft,hybrid=False,powermode="batt-dischrg"):
        exec parse_variables(Climb.__doc__)
        self.flightstate = FlightState()
        perf = aircraft.dynamic(self.flightstate,hybrid,powermode=powermode,t_charge = t)
        self.perf = perf
        CL = self.CL = perf.bw_perf.C_L
        S = self.S = aircraft.bw.wing["S"]
        CD = self.CD = perf.CD
        V = perf.fs.V
        rho = perf.fs.rho

        constraints = [
            perf.batt_perf.P <= aircraft.battery.m*aircraft.battery.P_max_cont,
            perf.bw_perf.eta_prop == 0.87,
            W ==  aircraft.mass*g,
            perf.bw_perf.C_T*rho*S*V**2 >= 0.5*CD*rho*S*V**2 + W*h_dot/V,
            h_gain <= h_dot*t,
            Sclimb == V*t, #sketchy constraint, is wrong with cos(climb angle)
        ]
        return constraints, perf

class Cruise(Model):
    """

    Variables
    ---------
    R           [nmi]       Range flown in flight segment
    t           [min]       Time to fly flight segment
    Vmin  120   [kts]       minimum cruise speed
    """

    def setup(self,aircraft,hybrid=False):
        exec parse_variables(Cruise.__doc__)
        self.flightstate = FlightState()
        self.perf = aircraft.dynamic(self.flightstate,hybrid,powermode="batt-chrg",t_charge=t)
        constraints = [R == t*self.flightstate.V,
                       self.flightstate["V"] >= Vmin,
                       self.perf.bw_perf.C_LC == 0.534,
                       self.perf.bw_perf.eta_prop == 0.87]

        return constraints, self.flightstate, self.perf

class Reserve(Model):
    """

    Variables
    ---------
    t     30    [min]       Time to fly flight segment
    Vmin        [kts]       Minimum flight speed
    """

    def setup(self,aircraft,hybrid=False):
        exec parse_variables(Reserve.__doc__)
        self.flightstate = FlightState()
        self.perf = aircraft.dynamic(self.flightstate,hybrid,powermode="batt-chrg",t_charge=t)
        constraints = [self.perf.bw_perf.C_LC == 0.534,
                       self.perf.bw_perf.eta_prop == 0.87]

        return constraints, self.perf
     
class Landing(Model):
    """ landing model

    Variables
    ---------
    g           9.81        [m/s**2]    gravitational constant
    h_obst      50          [ft]        obstacle height
    Xgr                     [ft]        landing ground roll
    Xa                      [ft]        approach distance
    Xro                     [ft]        round out distance
    Xdec                    [ft]        deceleration distance
    msafety     1.4         [-]         landing safety margin
    tang        -0.0524     [-]         tan(gamma)
    cosg        0.9986      [-]         cos(gamma)
    sing        -0.0523     [-]         sin(gamma)
    h_r                     [ft]        height of roundout
    T_a                     [N]         approach thrust
    W                       [N]         aircraft weight
    r                       [ft]        radius of roundout maneuver
    Vs                      [kts]       stall velocity
    nz           1.25       [-]         load factor
    Xla                     [ft]        total landing distance                        
    mu_b         0.6        [-]         braking friction coefficient
    Sgr                     [ft]        landing distance
    mstall       1.2        [-]         stall margin on landing
    t                       [s]         time of landing maneuver
    """
    def setup(self, aircraft,hybrid=False):
        exec parse_variables(Landing.__doc__)

        fs = FlightState()

        S = self.S = aircraft.bw.wing["S"]
        rho = fs.rho
        perf = aircraft.dynamic(fs,hybrid,powermode="batt-dischrg")
        CL = perf.bw_perf.C_L
        CD = perf.bw_perf.C_D
        V = perf.fs.V
        rho = perf.fs.rho
        C_T = perf.bw_perf.C_T
        self.perf = perf
        with gpkit.SignomialsEnabled():
            constraints = [
                # perf.bw_perf.C_LC == 2.19,
                W == aircraft.mass*g,
                C_T >= CD, #+ (W*sing)/(0.5*rho*S*V**2),
                V == Vs*mstall,
                (Vs*mstall)**2  >= (2.*aircraft.mass*g/rho/S/CL),
                Xgr*(2*g*(mu_b)) >= (mstall*Vs)**2,
                Xla >= Xgr,
                Sgr >= Xla,
                t >= Sgr/(0.3*V),
                perf.bw_perf.eta_prop == 0.75,

            ]

        return constraints, fs,perf      



class Mission(Model):
    """ Mission

    Variables
    ---------
    Srunway         150         [ft]        runway length
    Sobstacle                   [ft]        obstacle length
    mrunway         1.4         [-]         runway margin
    mobstacle       1.4         [-]         obstacle margin
    R                           [nmi]       mission range
    Vstall          61          [kts]       power off stall requirement
    Vs                          [kts]       power off stall speed
    CLstall         2.5         [-]         power off stall CL
    dV                          [m/s]       dV
    CJmax                       [-]         maximum CJ of mission
    CLmax                       [-]         maximum CL of mission
    """
    def setup(self,poweredwheels=False,n_wheels=3,hybrid=False,perf=False):
        exec parse_variables(Mission.__doc__)
        self.aircraft = Aircraft(poweredwheels,n_wheels,hybrid)
        with Vectorize(4):
            self.takeoff = TakeOff(self.aircraft,poweredwheels,n_wheels,hybrid)
        self.obstacle_climb = Climb(self.aircraft,hybrid)
        self.climb = Climb(self.aircraft,hybrid,powermode="batt-chrg")
        self.cruise = Cruise(self.aircraft,hybrid)
        self.reserve = Reserve(self.aircraft,hybrid)
        self.landing = Landing(self.aircraft,hybrid)
        
        Wcent = Variable("W_{cent}","lbf","center aircraft weight")
        loading = self.aircraft.loading(self.cruise.flightstate,Wcent)

        self.fs = [self.takeoff,self.obstacle_climb,self.climb,self.cruise,self.reserve, self.landing]
        state = FlightState()
        with gpkit.SignomialsEnabled():
            constraints = [
                            self.aircraft.htail.Vh >= 0.001563*CJmax*CLmax + 0.0323*CLmax + 0.03014*CJmax + 0.5216,
                            CLmax >= self.takeoff.perf.bw_perf.C_L,
                            CLmax >= self.landing.perf.bw_perf.C_L,
                            CJmax >= self.takeoff.perf.bw_perf.C_J,
                            CJmax >= self.landing.perf.bw_perf.C_J,
                            self.obstacle_climb.h_gain == Variable("h_obstacle",50,"ft"),
                            self.climb.h_gain == Variable("h_cruise",1950,"ft"),
                            self.climb.Sclimb == Variable("Scruiseclimb",10,"miles"),
                            0.5*state.rho*CLstall*self.aircraft.bw.wing.planform.S*Vs**2 == self.aircraft.mass*g,
                            Vs <= Vstall,
                            self.takeoff.dV == dV,
                            # self.takeoff.Vf[0] == self.takeoff.dV,
                            (self.takeoff.fs.V[-1]/self.takeoff.mstall)**2 >= (2*self.aircraft.mass*g/(self.takeoff.rho*self.takeoff.S*self.takeoff.perf.bw_perf.C_L[-1])),
                            0.5*self.takeoff.perf.bw_perf.C_L[-1]*self.takeoff.perf.fs.rho*self.aircraft.bw.wing["S"]*self.takeoff.fs.V[-1]**2 >= self.aircraft.mass*g,
                            self.takeoff.perf.bw_perf.C_L[0:-1] >= Variable("a",1e-4,"-","dum"),
                            Srunway >= mrunway*sum(self.takeoff.Sto),
                            #midpoint velocity displacement
                            # self.takeoff.perf.bw_perf.C_D[:-1] == self.takeoff.perf.bw_perf.C_D[-1],
                            self.takeoff.dV[0]*0.5*self.takeoff.t[0] == self.takeoff.Sto[0],
                            sum(self.takeoff.dV[:2])*0.5*self.takeoff.t[1] <= self.takeoff.Sto[1],
                            sum(self.takeoff.dV[:3])*0.5*self.takeoff.t[2] <= self.takeoff.Sto[2],
                            sum(self.takeoff.dV[:4])*0.5*self.takeoff.t[3] <= self.takeoff.Sto[3],
                            self.takeoff.fs.V[0] >= sum(self.takeoff.dV[:1]),
                            self.takeoff.fs.V[1] >= sum(self.takeoff.dV[:2]),
                            self.takeoff.fs.V[2] >= sum(self.takeoff.dV[:3]),
                            self.takeoff.fs.V[-1] <=  sum(self.takeoff.dV),
                            Srunway >= self.landing.Sgr*mrunway,
                            Sobstacle <= Srunway + Variable("obstacle_dist",100,"ft"),
                            Sobstacle >= mobstacle*(sum(self.takeoff.Sto)+ self.obstacle_climb.Sclimb),
                            loading.wingl["W"] == Wcent,
                            Wcent >= self.aircraft.mass*g,
                            # self.takeoff.perf.bw_perf.C_L == 8,
                            # self.obstacle_climb.perf.bw_perf.C_L == 4,
                            self.climb.perf.bw_perf.C_LC == 0.611,
                            self.climb.perf.bw_perf.P <=  self.aircraft.bw.n_prop*self.aircraft.bw.powertrain.P_m_sp_cont*self.aircraft.bw.powertrain.m,
                            self.cruise.perf.bw_perf.P <= self.aircraft.bw.n_prop*self.aircraft.bw.powertrain.P_m_sp_cont*self.aircraft.bw.powertrain.m
                        ]
        if not perf:
            constraints += [self.R == Variable("R_req",100,"nmi","range requirement")]
        if hybrid:
            constraints += [ self.takeoff.perf.gen_perf.P_fuel == self.obstacle_climb.perf.gen_perf.P_fuel,
                            self.obstacle_climb.perf.gen_perf.P_fuel == self.climb.perf.gen_perf.P_fuel,
                            self.climb.perf.gen_perf.P_fuel == self.cruise.perf.gen_perf.P_fuel,
                       self.cruise.perf.gen_perf.P_fuel == self.landing.perf.gen_perf.P_fuel,
                            self.aircraft.battery.E_capacity*0.8 >= self.takeoff.perf.bw_perf.P*self.takeoff.t + self.obstacle_climb.perf.bw_perf.P*self.obstacle_climb.t,
                            self.aircraft.tank.E >= sum(s.t*s.perf.gen_perf.P_fuel for s in self.fs)]
        else:
            constraints += [self.aircraft.battery.E_capacity*0.8 >= sum(s.t*s.perf.batt_perf.P for s in self.fs)]
        with gpkit.SignomialsEnabled():
            constraints += [R <= self.cruise.R]
        return constraints,self.aircraft,self.fs, loading

def writeSol(sol):
    with open('solve.txt', 'wb') as output:
        output.write(sol.table())

def writeWgt(sol,M):

        # output.write(str('{:10.4f}'.format(float(sol(M.aircraft.mass).magnitude))) + '\n')

        m_tot   = sol(M.aircraft.mass       )
        m_wing  = sol(M.aircraft.bw.wing.W/g).to("kg")
        m_htail = sol(M.aircraft.htail.W/g  ).to("kg")
        m_vtail = sol(M.aircraft.vtail.W/g  ).to("kg")
        m_boom  = sol(M.aircraft.boom.W/g   ).to("kg")
        m_fuse  = sol(M.aircraft.fuselage.m ).to("kg")
        m_cabin = sol(M.aircraft.cabin.m    )
        m_equip = sol(M.aircraft.equipment.m).to("kg")
        m_gear  = sol(M.aircraft.gear.m     ).to("kg")
        m_batt  = sol(M.aircraft.battery.m  )
        m_fuel  = sol(M.aircraft.tank.m_fuel).to("kg")
        m_tank  = sol(M.aircraft.tank.m_tank)
        m_gen   = sol(M.aircraft.genandic.m )
        m_mot   = sol(M.aircraft.bw.powertrain.m     )
        m_pax   = sol(M.aircraft.mpax       )
        m_bag   = sol(M.aircraft.mbaggage   )
        n_prop  = sol(M.aircraft.bw.n_prop  )
        n_pax   = sol(M.aircraft.n_pax      )
        m_mot_tot = n_prop * m_mot
        m_pax_tot = n_pax  * m_pax
        m_bag_tot = n_pax  * m_bag

        output  = open('weights.csv','wb')

        output.write(str(m_tot  .magnitude) + '\n')
        output.write(str(m_wing .magnitude) + '\n')
        output.write(str(m_htail.magnitude) + '\n')
        output.write(str(m_vtail.magnitude) + '\n')
        output.write(str(m_boom .magnitude) + '\n')
        output.write(str(m_fuse .magnitude) + '\n')
        output.write(str(m_cabin.magnitude) + '\n')
        output.write(str(m_equip.magnitude) + '\n')
        output.write(str(m_gear .magnitude) + '\n')
        output.write(str(m_batt .magnitude) + '\n')
        output.write(str(m_fuel .magnitude) + '\n')
        output.write(str(m_tank .magnitude) + '\n')
        output.write(str(m_gen  .magnitude) + '\n')
        output.write(str(m_mot_tot.magnitude) + '\n')
        output.write(str(m_pax_tot.magnitude) + '\n')
        output.write(str(m_bag_tot.magnitude) + '\n')

        output.close
        
        output  = open('weights.txt','wb')

        output.write('Mass Summary\n')
        output.write('individual motor and passenger-related masses shown\n\n')
        output.write('m_tot   = ' + str(m_tot  ) + '\n')
        output.write('m_wing  = ' + str(m_wing ) + '\n')
        output.write('m_htail = ' + str(m_htail) + '\n')
        output.write('m_vtail = ' + str(m_vtail) + '\n')
        output.write('m_boom  = ' + str(m_boom ) + '\n')
        output.write('m_fuse  = ' + str(m_fuse ) + '\n')
        output.write('m_cabin = ' + str(m_cabin) + '\n')
        output.write('m_equip = ' + str(m_equip) + '\n')
        output.write('m_gear  = ' + str(m_gear ) + '\n')
        output.write('m_batt  = ' + str(m_batt ) + '\n')
        output.write('m_fuel  = ' + str(m_fuel ) + '\n')
        output.write('m_tank  = ' + str(m_tank ) + '\n')
        output.write('m_gen   = ' + str(m_gen  ) + '\n')
        output.write('m_mot   = ' + str(m_mot  ) + '\n')
        output.write('m_pax   = ' + str(m_pax  ) + '\n')
        output.write('m_bag   = ' + str(m_bag  ) + '\n')
        output.write('n_prop  = ' + str(n_prop ) + ' [-]' + '\n')
        output.write('n_pax   = ' + str(n_pax  ) + ' [-]' + '\n')

        output.close

def writeProp(sol,M):

        output  = open('prop.txt','wb')
        
        A_str = '{:7.4f}'
        T_str = '{:9.2f}'
        V_str = '{:7.2f}'
        
        A_disk = sol(M.cruise.perf.bw_perf.A_disk).magnitude
        
        T_TO1 = sol(M.takeoff.perf.bw_perf.T[0]    ).magnitude
        T_TO2 = sol(M.takeoff.perf.bw_perf.T[1]    ).magnitude
        T_TO3 = sol(M.takeoff.perf.bw_perf.T[2]    ).magnitude
        T_TO4 = sol(M.takeoff.perf.bw_perf.T[3]    ).magnitude
        T_CL1 = sol(M.obstacle_climb.perf.bw_perf.T).magnitude
        T_CL2 = sol(M.climb.perf.bw_perf.T         ).magnitude
        T_CR  = sol(M.cruise.perf.bw_perf.T        ).magnitude
        T_R   = sol(M.reserve.perf.bw_perf.T       ).magnitude
        T_L   = sol(M.landing.perf.bw_perf.T       ).magnitude
        
        V_TO1 = sol(M.takeoff.perf.fs.V[0]    ).to("m/s").magnitude
        V_TO2 = sol(M.takeoff.perf.fs.V[1]    ).to("m/s").magnitude
        V_TO3 = sol(M.takeoff.perf.fs.V[2]    ).to("m/s").magnitude
        V_TO4 = sol(M.takeoff.perf.fs.V[3]    ).to("m/s").magnitude
        V_CL1 = sol(M.obstacle_climb.perf.fs.V).to("m/s").magnitude
        V_CL2 = sol(M.climb.perf.fs.V         ).to("m/s").magnitude
        V_CR  = sol(M.cruise.perf.fs.V        ).to("m/s").magnitude
        V_R   = sol(M.reserve.perf.fs.V       ).to("m/s").magnitude
        V_L   = sol(M.landing.perf.fs.V       ).to("m/s").magnitude
        
        u_j_TO1 = sol(M.takeoff.perf.bw_perf.u_j[0]    ).to("m/s").magnitude
        u_j_TO2 = sol(M.takeoff.perf.bw_perf.u_j[1]    ).to("m/s").magnitude
        u_j_TO3 = sol(M.takeoff.perf.bw_perf.u_j[2]    ).to("m/s").magnitude
        u_j_TO4 = sol(M.takeoff.perf.bw_perf.u_j[3]    ).to("m/s").magnitude
        u_j_CL1 = sol(M.obstacle_climb.perf.bw_perf.u_j).to("m/s").magnitude
        u_j_CL2 = sol(M.climb.perf.bw_perf.u_j         ).to("m/s").magnitude
        u_j_CR  = sol(M.cruise.perf.bw_perf.u_j        ).to("m/s").magnitude
        u_j_R   = sol(M.reserve.perf.bw_perf.u_j       ).to("m/s").magnitude
        u_j_L   = sol(M.landing.perf.bw_perf.u_j       ).to("m/s").magnitude
        
        output.write('A_disk = ' + str(A_str.format(float(A_disk))) + ' m^2' + '\n\n')

        output.write('       T_tot [N]  V [m/s]  uj [m/s]')
        output.write('\n' + 'TO1 = ' + str(T_str.format(float(T_TO1)))
                          + '  '     + str(V_str.format(float(V_TO1)))
                          + '  '     + str(V_str.format(float(u_j_TO1))))
        output.write('\n' + 'TO2 = ' + str(T_str.format(float(T_TO2)))
                          + '  '     + str(V_str.format(float(V_TO2)))
                          + '  '     + str(V_str.format(float(u_j_TO2))))
        output.write('\n' + 'TO3 = ' + str(T_str.format(float(T_TO3)))
                          + '  '     + str(V_str.format(float(V_TO3)))
                          + '  '     + str(V_str.format(float(u_j_TO3))))
        output.write('\n' + 'TO4 = ' + str(T_str.format(float(T_TO4)))
                          + '  '     + str(V_str.format(float(V_TO4)))
                          + '  '     + str(V_str.format(float(u_j_TO4))))
        output.write('\n' + 'CL1 = ' + str(T_str.format(float(T_CL1)))
                          + '  '     + str(V_str.format(float(V_CL1)))
                          + '  '     + str(V_str.format(float(u_j_CL1))))
        output.write('\n' + 'CL2 = ' + str(T_str.format(float(T_CL2)))
                          + '  '     + str(V_str.format(float(V_CL2)))
                          + '  '     + str(V_str.format(float(u_j_CL2))))
        output.write('\n' + 'CR  = ' + str(T_str.format(float(T_CR )))
                          + '  '     + str(V_str.format(float(V_CR )))
                          + '  '     + str(V_str.format(float(u_j_CR ))))
        output.write('\n' + 'R   = ' + str(T_str.format(float(T_R  )))
                          + '  '     + str(V_str.format(float(V_R  )))
                          + '  '     + str(V_str.format(float(u_j_R  ))))
        output.write('\n' + 'L   = ' + str(T_str.format(float(T_L  )))
                          + '  '     + str(V_str.format(float(V_L  )))
                          + '  '     + str(V_str.format(float(u_j_L  ))))

        output.close


def writeAlb(sol,M):
    with open('albie.txt', 'wb') as output:
        output.write('Weights Summary\n')
        # output.write(sol(M.aircraft.mass).split(" ")[0])
        # output.write(sol.table(["m_Mission/Aircraft/Fuselage",
                                # "m_Mission/Aircraft/Cabin",
                                # "W_Mission/Aircraft/HorizontalTail",
                                # "W_Mission/Aircraft/VerticalTail",
                                # "W_Mission/Aircraft/TailBoom",
                                # "W_Mission/Aircraft/BlownWing/Wing",
                                # "n_prop_Mission/Aircraft/BlownWing",
                                # "m_Mission/Aircraft/BlownWing/Powertrain",
                                # "m_Mission/Aircraft/Gear",
                                # "m_Mission/Aircraft/GenAndIC",
                                # "m_fuel_Mission/Aircraft/Tank",
                                # "m_tank_Mission/Aircraft/Tank",
                                # "n_pax_Mission/Aircraft",
                                # "mbaggage_Mission/Aircraft",
                                # "mpax_Mission/Aircraft",
                                # "m_Mission/Aircraft/Battery"
                                # ]))
        output.write('\n\n\nPropulsion Summary')


def CLCurves():
    M = Mission(poweredwheels=True,n_wheels=3,hybrid=True)
    range_sweep = np.linspace(115,600,4)
    M.substitutions.update({M.R:('sweep',range_sweep)})
    M.cost = M.aircraft.mass
    sol = M.localsolve("mosek")
    print sol.summary()
    plt.plot(sol(M.R),sol(M.aircraft.mass))

    # sol = M.solve()
    # # CLmax_set = np.linspace(3.5,8,5)
    # for CLmax in CLmax_set:
    #     print CLmax
    #     # M.substitutions.update({M.aircraft.bw.CLmax:CLmax})
    #     sol = M.solve("mosek")
    #     print sol(M.aircraft.mass)
    
    plt.grid()
    # plt.xlim([0,300])
    # plt.ylim([0,1600])
    plt.title("Impact of range on takeoff mass")
    plt.xlabel("Cruise segment range [mi]")
    plt.ylabel("Takeoff mass [kg]")
    # plt.legend()
    plt.show()

def RangeRunway():
    M = Mission(poweredwheels=False,n_wheels=3,hybrid=True)
    range_sweep = np.linspace(100,300,8)
    M.substitutions.update({M.R:('sweep',range_sweep)})
    M.cost = M.aircraft.mass
    # sol = M.localsolve("mosek")
    # print sol.summary()

    # M.substitutions.update({M.Srunway:s})
    sol = M.localsolve("mosek")
    plt.plot(sol(M.R),sol(M.aircraft.mass))

    # plt.plot(sol(M.R),sol(M.aircraft.mass))
    plt.grid()
    # plt.xlim([0,300])
    plt.ylim(ymin=0)
    plt.title("Impact of range on takeoff mass")
    plt.xlabel("Cruise segment range [nmi]")
    plt.ylabel("Takeoff mass [kg]")
    # plt.legend()
    plt.show()

def RangeSpeed():
    M = Mission(poweredwheels=True,n_wheels=3,hybrid=True)
    range_sweep = np.linspace(115,415,4)
    M.substitutions.update({M.Srunway:100})
    M.substitutions.update({M.R:('sweep',range_sweep)})
    M.cost = M.aircraft.mass
    # sol = M.localsolve("mosek")
    # print sol.summary()

    speed_set = np.linspace(120,150,4)
    for s in speed_set:
        M.substitutions.update({M.cruise.Vmin:s})
        sol = M.localsolve("mosek")
        plt.plot(sol(M.R),sol(M.aircraft.mass),label=str(s)+ " kt")

    # plt.plot(sol(M.R),sol(M.aircraft.mass))
    plt.grid()
    # plt.xlim([0,300])
    # plt.ylim([0,1600])
    plt.title("Impact of range on takeoff mass")
    plt.xlabel("Cruise segment range [nmi]")
    plt.ylabel("Takeoff mass [kg]")
    plt.legend()
    plt.show()

def SpeedSweep():
    M = Mission(poweredwheels=True,n_wheels=3,hybrid=True)
    speed_sweep = np.linspace(120,160,5)
    M.substitutions.update({M.cruise.Vmin:('sweep',speed_sweep)})
    M.cost = M.aircraft.mass
    sol = M.localsolve("mosek")

    plt.plot(sol(M.cruise.Vmin),sol(M.aircraft.mass))
    plt.grid()
    # plt.xlim([0,300])
    # plt.ylim([0,1600])
    plt.title("Impact of speed on takeoff mass")
    plt.xlabel("Cruise segment speed [kts]")
    plt.ylabel("Takeoff mass [kg]")
    plt.legend()
    plt.show()

def ElectricVsHybrid():
    M = Mission(poweredwheels=False,n_wheels=3,hybrid=True)
    runway_sweep = np.linspace(80,300,10)
    M.substitutions.update({M.Srunway:('sweep',runway_sweep)})
    # M.substitutions.update({M.aircraft.bw.n_prop:6})
    M.cost = M.aircraft.mass
    sol = M.localsolve("mosek")
    print sol.summary()
    
    plt.plot(sol(M.Srunway),sol(M.aircraft.mass),label='mstall = 1.1')

    M = Mission(poweredwheels=False,n_wheels=3,hybrid=True)
    # M.substitutions.update({M.aircraft.battery.:1.2})
    # runway_sweep = np.linspace(300,1000,8)
    M.substitutions.update({M.Srunway:('sweep',runway_sweep)})
    # M.substitutions.update({M.aircraft.bw.n_prop:6})
    M.cost = M.aircraft.mass
    sol = M.localsolve("mosek")

    plt.plot(sol(M.Srunway),sol(M.aircraft.mass),label='mstall = 1.2')
    plt.legend()
    plt.title("Stall margin tradeoff")
    plt.xlabel("Runway length [ft]")
    plt.ylabel("Takeoff mass [kg]")
    plt.grid()
    plt.show()

def ICVsTurboshaft():
    M = Mission(poweredwheels=True,n_wheels=3,hybrid=True)
    runway_sweep = np.linspace(150,300,5)
    M.substitutions.update({M.Srunway:('sweep',runway_sweep)})
    M.cost = M.aircraft.mass
    sol = M.localsolve("mosek")
    print sol.summary()
    
    plt.plot(sol(M.Srunway),sol(M.aircraft.mass),label='Powered wheels')

    M = Mission(poweredwheels=False,n_wheels=3,hybrid=True)
    runway_sweep = np.linspace(150,300,5)
    M.substitutions.update({M.Srunway:('sweep',runway_sweep)})
    # M.substitutions.update({M.aircraft.genandic.P_ic_sp_cont:1})
    # M.substitutions.update({M.aircraft.genandic.eta_IC:0.2656})
    M.cost = M.aircraft.mass
    sol = M.localsolve("mosek")

    plt.plot(sol(M.Srunway),sol(M.aircraft.mass),label='No powered wheels')
    plt.legend()
    plt.title("Powered wheels trade")
    plt.xlabel("Runway length [ft]")
    plt.ylabel("Takeoff mass [kg]")
    plt.grid()
    plt.ylim(ymin=0)
    plt.show()

def Runway():
    M = Mission(poweredwheels=False,n_wheels=3,hybrid=True)
    runway_sweep = np.linspace(100,300,10)
    M.substitutions.update({M.Srunway:('sweep',runway_sweep)})
    M.cost = M.aircraft.mass
    sol = M.localsolve("mosek")
    print sol.summary()

    plt.plot(sol(M.Srunway),sol(M.aircraft.mass))
    plt.title("Runway trade")
    plt.xlabel("Runway length [ft]")
    plt.ylabel("Takeoff mass [kg]")
    plt.ylim(ymin=0)
    plt.grid()
    plt.show()    

def PerfPlot():
    poweredwheels = False

    M = Mission(poweredwheels=poweredwheels,n_wheels=3,hybrid=True,perf=False)
    M.cost = M.aircraft.mass
    # M.debug()
    sol = M.localsoslve("mosek")
    # print M.program.gps[-1].result.summary()
    print sol.summary()

    print "fixed solve"
    M2 = Mission(poweredwheels=poweredwheels,n_wheels=3,hybrid=True,perf=True)
    #Fix airplane design
    M2.substitutions.update({M2.aircraft.mass:sol(M.aircraft.mass)})
    M2.substitutions.update({M2.aircraft.bw.wing.planform.AR:sol(M.aircraft.bw.wing.planform.AR)})
    M2.substitutions.update({M2.aircraft.bw.powertrain.m:sol(M.aircraft.bw.powertrain.m)})

    for n_pax in [1,2,3,4]:
        M2.substitutions.update({M2.Srunway:('sweep',np.linspace(300,500,4))})
        M2.substitutions.update({M2.aircraft.n_pax:n_pax})
        M2.cost = 1/M2.R
        sol2 = M2.localsolve("mosek")
        plt.plot(sol2(M2.Srunway),sol2(M2.R),label = str(n_pax-1) + " passengers")
    
    plt.title("Performance Trade for 300 ft Design")
    plt.xlabel("Runway length [ft]")
    plt.ylabel("Range [nmi]")
    plt.legend()
    plt.ylim(ymin=0)
    plt.grid()
    plt.show()    
    # print sol2.table()
    # sd = get_highestsens(M, sol, N=10)
    # f, a = plot_chart(sd)
    # f.savefig("sensbar.pdf", bbox_inches="tight")
    # print sol(M.aircraft.mass)
    # writeSol(sol)
    # writeAlb(sol,M)

def PowerIncrease():
    poweredwheels = False
    M = Mission(poweredwheels=poweredwheels,n_wheels=3,hybrid=True,perf=False)
    M.cost = M.aircraft.mass
    sol = M.localsolve("mosek")
    print sol.summary()

    print "fixed solve"
    M2 = Mission(poweredwheels=poweredwheels,n_wheels=3,hybrid=True,perf=False)
    #Fix airplane design
    # M2.substitutions.update({M2.aircraft.mass:sol(M.aircraft.mass)})
    M2.substitutions.update({M2.aircraft.bw.wing.planform.S:sol(M.aircraft.bw.wing.planform.S)})    
    M2.substitutions.update({M2.aircraft.bw.wing.planform.AR:sol(M.aircraft.bw.wing.planform.AR)})
    M2.substitutions.update({M2.aircraft.htail.planform.AR:sol(M.aircraft.htail.planform.AR)})

    # M2.substitutions.update({M2.aircraft.bw.powertrain.m:sol(M.aircraft.bw.powertrain.m)})
    pbs = [180,64,"PBS TS-100DA"]
    rolls = [313,73, "RR M250-C20B/F/J"]
    for i,eng in enumerate([pbs]):
        M2.substitutions.update({M2.Srunway:('sweep',np.linspace(150,300,4))})
        M2.substitutions.update({M2.aircraft.genandic.P_turb_sp_cont:eng[0]/eng[1]})
        M2.substitutions.update({M2.aircraft.genandic.m_turb:eng[1]})
        M2.cost = 1/M2.aircraft.mass
        sol2 = M2.localsolve("mosek")
        plt.plot(sol2(M2.Srunway),0.5*sol2(M2.aircraft.mass),label = eng[2])

    plt.title("Power increase trade")
    plt.xlabel("Runway length [ft]")
    plt.ylabel("Mass [kg]")
    plt.legend()
    plt.ylim(ymin=0)
    plt.grid()
    plt.show() 


def RegularSolve():
    poweredwheels = False
    M = Mission(poweredwheels=poweredwheels,n_wheels=3,hybrid=True)
    M.cost = M.aircraft.mass
    # M.debug()
    sol = M.localsolve("mosek")
    # print M.program.gps[-1].result.summary()
    print sol.summary()
    sd = get_highestsens(M, sol, N=10)
    f, a = plot_chart(sd)
    f.savefig("sensbar.pdf", bbox_inches="tight")
    print sol(M.aircraft.mass)
    print sol["sensitivities"]["constants"]["CLCmax"]
    writeSol(sol)
    writeAlb(sol,M)
    writeProp(sol,M)
    writeWgt(sol,M)
# CLCurves()
# RangeRunway()
# RangeSpeed()
# SpeedSweep()
# ElectricVsHybrid()
# ICVsTurboshaft()

if __name__ == "__main__":
    # Runway()
    # RangeRunway()
    RegularSolve()
    # PerfPlot()
    # PowerIncrease()
    # ElectricVsHybrid()

