import os
import pandas as pd
from gpkit import Model, parse_variables
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
from decimal import *
pi = math.pi

class Aircraft(Model):
    """ Aircraft

    Variables
    ---------
    m               [kg]    aircraft mass
    Npax        4   [-]     number of passengers
    mpax        93  [kg]    mass of a passenger
    mbaggage    9   [kg]    mass of baggage
    """
    def setup(self,poweredwheels=False,n_wheels=3,hybrid=False):
        exec parse_variables(Aircraft.__doc__)

        self.battery = Battery()
        self.fuselage = Fuselage()
        self.bw = BlownWing()
        self.cabin = Cabin()
        self.htail = HorizontalTail()
        self.vtail = VerticalTail()
        self.vtail.substitutions[self.vtail.planform.tau] = 0.08
        self.htail.substitutions[self.htail.planform.tau] = 0.08
        self.htail.substitutions[self.htail.mh] = 0.8
        # self.htail.substitutions[self.htail.Vh] = 0.4

        self.components = [self.cabin,self.bw,self.battery,self.fuselage]
    
        if hybrid:
            self.tank = Tank()
            self.genandic = GenAndIC()
            self.components += [self.tank,self.genandic]

        if poweredwheels:
                self.pw = PoweredWheel()
                self.wheels = n_wheels*[self.pw]
                self.components += self.wheels
        self.mass = m
        constraints = [

                       # self.htail.Vh <= (self.htail["S"]*self.htail.lh/self.bw.wing["S"]**2 *self.bw.wing["b"]),
                       # self.vtail.Vv == (self.vtail["S"]*self.vtail.lv/self.bw.wing["S"]/self.bw.wing["b"]),
                       self.vtail.planform["b"] >= Variable("bv",48,"in"),
                       self.vtail.planform["croot"] >= Variable("croot",27,"in"),

                       self.htail.planform["b"] >= Variable("bh",60,"in"),
                       self.htail.planform["croot"] >= Variable("croot",27,"in"),

                       self.vtail.lv == Variable("lv",180,"in"),
                       self.htail.lh == Variable("lh",180,"in"),

                       self.fuselage.m >= 0.4*(sum(c.topvar("m") for c in self.components) + (self.vtail.W + self.htail.W)/g),
                       self.mass>=sum(c.topvar("m") for c in self.components) + (self.vtail.W + self.htail.W)/g+ (mpax+mbaggage)*Npax]

        with gpkit.SignomialsEnabled():
            constraints += [self.bw.wing.b - Variable("w_fuse",50,"in") >= self.bw.n_prop*2*self.bw.powertrain.r]
        return constraints, self.components, self.htail, self.vtail

    def dynamic(self,state,hybrid=False,powermode="batt-chrg",t_charge=None):
        return AircraftP(self,state,hybrid,powermode=powermode,t_charge=t_charge)
    def loading(self,Wcent,state):
        return AircraftLoading(self,state)

class AircraftP(Model):
    """ AircraftP

    Variables
    ---------
    P           [kW]    total power draw
    CD          [-]     total CD, referenced to wing area
    P_charge    [kW]    battery charging power
    """
    def setup(self,aircraft,state,hybrid=False,powermode="batt-chrg",t_charge=None):
        exec parse_variables(AircraftP.__doc__)
        self.bw_perf = aircraft.bw.dynamic(state)
        self.batt_perf = aircraft.battery.dynamic(state)
        self.htail_perf = aircraft.htail.flight_model(aircraft.htail, state)
        self.vtail_perf = aircraft.vtail.flight_model(aircraft.vtail, state)
        self.fuse_perf = aircraft.fuselage.dynamic(state)
        self.perf_models = [self.bw_perf,self.batt_perf,self.htail_perf,self.vtail_perf,self.fuse_perf]
        self.fs = state
        constraints = [0.5*self.bw_perf.C_L*state.rho*aircraft.bw.wing["S"]*state.V**2 >= aircraft.mass*state["g"],
                       P >= self.bw_perf["P"],
                       # self.batt_perf.P >= P,
                       CD >= self.bw_perf.C_D + (aircraft.fuselage.Swet/aircraft.bw.wing.planform.S)*self.fuse_perf.Cd + ((aircraft.htail.planform.S/aircraft.bw.wing.planform.S)*self.htail_perf.Cd + (aircraft.vtail.planform.S/aircraft.bw.wing.planform.S)*self.vtail_perf.Cd),
                       self.bw_perf.C_T >= CD
                    ]
        if hybrid:
            self.gen_perf = aircraft.genandic.dynamic(state)
            if powermode == "batt-chrg":
                constraints += [P_charge >= aircraft.battery.E_capacity/t_charge,
                                self.gen_perf.P_out >= P + P_charge,
                                self.batt_perf.P >= Variable("P_draw_batt",1e-4,"W")]
            if powermode == "batt-dischrg":
                with gpkit.SignomialsEnabled():
                    constraints += [self.batt_perf.P + self.gen_perf.P_out >= P]
            self.perf_models += [self.gen_perf]
        return constraints,self.perf_models

class AircraftLoading(Model):
    def setup(self,aircraft,state):
        self.wingl = aircraft.bw.wing.spar.loading(aircraft.bw.wing, state)
        loading = [self.wingl]
        return loading

class Cabin(Model):
    """Cabin
    Variables
    ---------
    m        78.43     [kg]       total mass
    """
    def setup(self):
        exec parse_variables(Cabin.__doc__)
        return []
class Fuselage(Model):
    """ Fuselage

    Variables
    ---------
    m                   [kg]    mass of fuselage
    l       270         [in]    length
    Swet    29833.67    [in^2]     wetted area of fuselage    
    """
    def setup(self):
        exec parse_variables(Fuselage.__doc__)
    def dynamic(self,state):
        return FuselageP(self,state)

class FuselageP(Model):
    """FuselageP
    Variables
    ---------
    Cd              [-]     drag coefficient
    """
    def setup(self,fuse,state):
        exec parse_variables(FuselageP.__doc__)
        constraints = [Cd >= 0.455/((state.rho*state.V*fuse.l/state.mu)**0.3)]
        return constraints

class Powertrain(Model):
    """ Powertrain
    Variables
    ---------
    m                       [kg]        powertrain mass
    m_m                     [kg]        motor mass
    m_mc                    [kg]        motor controller mass
    Pmax                    [kW]        maximum power
    P_m_sp_cont   7         [kW/kg]      motor specific power
    P_mc_sp_cont  11.8      [kW/kg]     motor controller specific power
    r                       [m]         propeller radius
    Pstar_ref     1         [W]         specific motor power reference
    m_ref         1         [kg]        reference motor power
    """

    def setup(self):
        exec parse_variables(Powertrain.__doc__)
                       

        constraints = [#P_m_sp_cont <= Pstar_ref*(-0.228*(m_m/m_ref)**2+45.7*(m_m/m_ref)+3060),
                        m >= m_m+m_mc,
                        Pmax <= m_m*P_m_sp_cont,
                        Pmax <= m_mc*P_mc_sp_cont]
        return constraints

class PoweredWheel(Model):
    """Powered Wheels

    Variables
    ---------
    RPMmax                  [rpm]       maximum RPM of motor
    gear_ratio              [-]         gear ratio of powered wheel
    gear_ratio_max  20      [-]         max gear ratio of powered wheel
    tau_max                 [N*m]       torque of the
    m                       [kg]        mass of powered wheel motor
    m_ref           1       [kg]        reference mass for equations
    r               0.2     [m]         tire radius
    Pstar           5       [kW/kg]     specific power for powered wheel
    """
    def setup(self):
        exec parse_variables(PoweredWheel.__doc__)
        #Currently using the worst values
        with gpkit.SignomialsEnabled():
            constraints = [gear_ratio <= gear_ratio_max,
                           RPMmax <= Variable("a",4.9,"rpm/kg^2")*m**2 - Variable("b",313.3,"rpm/kg")*m +Variable("c",8721.2,"rpm"),
                           tau_max <= Variable("d",27.3,"N*m/kg")*m - Variable("e",80.2,"N*m"),
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
    P_ic_sp_cont    1              [kW/kg]     specific cont power of IC
    eta_IC          0.256          [-]        thermal efficiency of IC
    m_g                            [kg]       genandic mass
    m_gc                           [kg]       genandic controller mass
    m_ic                           [kg]       piston mass
    P_g_sp_cont                    [W/kg]     genandic spec power (cont)
    P_g_cont                       [W]        genandic cont. power
    P_gc_cont                      [W]        genandic controller cont. power
    P_gc_sp_cont   11.8            [kW/kg]     genandic controller cont power
    P_ic_cont                      [W]        piston continous power  
    m                              [kg]       total mass
    m_ref           1              [kg]       reference mass, for meeting units constraints
    Pstar_ref       1              [W/kg]     reference specific power, for meeting units constraints
    """
    def setup(self):
        exec parse_variables(GenAndIC.__doc__)
        with gpkit.SignomialsEnabled():
            constraints = [P_g_sp_cont/Pstar_ref <= -0.228*(m_g/m_ref)**2+45.7*(m_g/m_ref)+3060,
                           P_g_cont    <=   P_g_sp_cont*m_g,
                           P_gc_cont   <=   P_gc_sp_cont*m_gc,
                           P_ic_cont   <=   P_ic_sp_cont*m_ic,
                           m >= m_g + m_gc + m_ic
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
    P_ic                            [kW]        internal combustion power
    P_fuel                          [kW]        power coming in from fuel flow
    P_out                           [kW]        output from generator controller after efficiency
    eta_wiring      0.98            [-]         efficiency of electrical connections (wiring loss)
    eta_gc          0.98            [-]         efficiency of generator controller
    eta_shaft       0.98            [-]         shaft losses (two 99% efficient bearings)
    eta_g           0.9             [-]         generator efficiency
    eta_ic          0.256           [-]         internal combustion engine efficiency
    """
    def setup(self,gen,state):
        exec parse_variables(genandicP.__doc__)
        constraints = [P_g <= gen.P_g_cont,
                       P_gc <= gen.P_gc_cont,
                       P_ic <= gen.P_ic_cont,
                       P_fuel*eta_ic == P_ic,
                       P_ic*eta_shaft == P_g,
                       P_g*eta_g == P_gc,
                       P_gc*eta_gc == P_out 
                       ]
        return constraints

class Tank(Model):
    """Tank Model
    Variables
    ---------
    m                          [lb]          total mass
    m_fuel                     [lb]          mass of fuel
    E                          [Wh]          fuel energy
    Estar_fuel          11.9   [kWh/kg]      fuel specific energy
    V_fuel                     [gal]         fuel volume
    rho_fuel            6.7    [lb/gal]      fuel density
    m_tank                     [kg]          fuel tank mass
    rho_tank            0.55   [lb/gal]      empty tank mass (structure mass) per volume
    """
    def setup(self):
        exec parse_variables(Tank.__doc__)
        constraints = [V_fuel*rho_fuel >= m_fuel,
                       m_tank >= V_fuel*rho_tank,
                       m_fuel >= E/Estar_fuel,
                       m >= m_fuel+m_tank]
        return constraints


class Battery(Model):
    """ Battery
    Variables
    ---------
    m                   [kg]            total mass
    Estar       140     [Wh/kg]         specific energy
    E_capacity          [Wh]            energy capacity
    P_max_cont  4.2e3   [W/kg]          continuous power output
    P_max_burst 7e3     [W/kg]          burst power output
    """

    def setup(self):
        exec parse_variables(Battery.__doc__)
        with gpkit.SignomialsEnabled():
            constraints = [m >= E_capacity/Estar,
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
    n_prop     6    [-]             number of props
    m               [kg]            mass
    """
    def setup(self):
        exec parse_variables(BlownWing.__doc__)
        self.powertrain = Powertrain()
        N = 14
        self.wing = Wing(N)
        self.wing.substitutions[self.wing.planform.tau]=0.12

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
    mcdp    1.1     [-]             profile drag margin factor
    m_dotprime      [kg/(m*s)]      jet mass flow per unit span
    J_prime         [kg/(s**2)]      momentum flow per unit span
    E_prime         [J/(m*s)]       energy flow per unit span
    rho_j   1.225   [kg/m**3]        density in jet flow
    u_j             [kts]           velocity in jet flow
    h               [m]             Wake height
    T               [N]             propeller thrust
    P               [kW]            power draw
    eta_mc    0.98  [-]             motor controller efficiency
    eta_m     0.9   [-]             motor efficiency
    eta_prop  0.87  [-]             prop efficiency loss after blade disk actuator
    A_disk          [m**2]          area of prop disk
    Mlim      0.95  [-]             tip limit
    a         343   [m/s]           speed of sound at sea level
    k_t       0.2   [-]             propeller torque coefficient
    RPMmax          [rpm]           maximum rpm of propeller
    """
    def setup(self,bw,state):
        #bw is a BlownWing object
        #state is a FlightState
        exec parse_variables(BlownWingP.__doc__)
        with gpkit.SignomialsEnabled():
            constraints = [
            A_disk == bw.n_prop*pi*bw.powertrain.r**2,
            ((P*eta_prop*eta_m*eta_mc)/(0.5*T*state["V"]) - 1)**2 >= (T/(A_disk*(state.V**2)*state.rho/2)+1),
            (u_j/state.V)**2 <= (T/(A_disk*(state.V**2)*state.rho/2) + 1),
            u_j >= state.V,
            P <= bw.n_prop*bw.powertrain["Pmax"],
            C_L <= C_LC*(1+2*C_J/(pi*bw.wing["AR"]*e)),

            RPMmax >= Variable("a",4.9,"rpm/kg^2")*bw.powertrain.m_m**2 - Variable("b",313.3,"rpm/kg")*bw.powertrain.m_m +Variable("c",8721.2,"rpm"),
            RPMmax*bw.powertrain.r <= a*Mlim,

            C_T == T/((0.5*state.rho*bw.wing["S"]*state.V**2)),
            m_dotprime == rho_j*u_j*h,
            J_prime ==  m_dotprime*u_j,
            E_prime == 0.5*m_dotprime*u_j**2,
            C_Q ==  m_dotprime/(state.rho*state.V* bw.wing["cmac"]),
            C_J == J_prime/(0.5*state.rho*state.V**2 * bw.wing["cmac"]),
            C_E == E_prime/(0.5*state.rho*state.V**3 * bw.wing["cmac"]),
            h == pi*bw.powertrain.r/2,
            C_Di == (C_L**2)/(pi*bw.wing["AR"]*e),
            C_D >= C_Di  + C_Dp,
            C_Dp == mcdp*1.328/Re**0.5, #friction drag only, need to add form
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
    g           9.8         [m/s/s]         acceleration due to gravity
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
    mu_friction 0.8         [-]         traction limit for powered wheels
    t                       [s]         time of takeoff maneuver
    """
    def setup(self, aircraft,poweredwheels,n_wheels,hybrid=False):
        exec parse_variables(TakeOff.__doc__)

        fs = FlightState()

        path = os.path.dirname(os.path.abspath(__file__))
        df = pd.read_csv(path + os.sep + "logfit.csv")
        fd = df.to_dict(orient="records")[0] #fit data

        S = aircraft.bw.wing["S"]
        Pmax = aircraft.bw.powertrain.Pmax
        AR = aircraft.bw.wing.planform.AR
        rho = fs.rho
        V = fs.V
        perf = aircraft.dynamic(fs,hybrid,powermode="batt-dischrg")
        self.perf = perf
        e = perf.bw_perf.e
        mstall = 1.3
        constraints = [
                perf.bw_perf.C_LC == 2.18,
                W == aircraft.mass*fs.g,
                T/W >= A/g + mu,

                CDg >= perf.bw_perf.C_D,
                V >= mstall*(2*W/rho/S/perf.bw_perf.C_L)**0.5,
                FitCS(fd, zsto, [A/g, B*V**2/g]), #fit constraint set, pass in fit data, zsto is the 
                # y variable, then arr of independent (input) vars, watch the units
                Sto >= 1.0/2.0/B*zsto,
                t >= Sto/(0.3*V)]
        with gpkit.SignomialsEnabled():
            constraints += [B >= g/W*0.5*rho*S*(CDg)]
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

        return constraints, fs, perf

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

    def setup(self,aircraft,hybrid=False):
        exec parse_variables(Climb.__doc__)
        self.flightstate = FlightState()
        perf = aircraft.dynamic(self.flightstate,hybrid,powermode="batt-dischrg")
        self.perf = perf
        CL = self.CL = perf.bw_perf.C_L
        S = self.S = aircraft.bw.wing["S"]
        CD = self.CD = perf.CD
        V = perf.fs.V
        rho = perf.fs.rho

        constraints = [
            perf.batt_perf.P <= aircraft.battery.m*aircraft.battery.P_max_cont,
            W ==  aircraft.mass*perf.fs.g,
            W <= 0.5*CL*rho*S*V**2,
            perf.bw_perf.T >= 0.5*CD*rho*S*V**2 + W*h_dot/V,
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
    Vmin  120   [kts]       Minimum flight speed
    """

    def setup(self,aircraft,hybrid=False):
        exec parse_variables(Cruise.__doc__)
        self.flightstate = FlightState()
        self.perf = aircraft.dynamic(self.flightstate,hybrid,powermode="batt-chrg",t_charge=t)
        constraints = [R == t*self.flightstate.V,
                       self.flightstate["V"] >= Vmin,
                       self.perf.bw_perf.C_LC == 0.8]

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
    mu_b         0.8        [-]         braking friction coefficient
    Sgr                     [ft]        landing distance
    mstall       1.3        [-]         stall
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
                perf.bw_perf.C_LC == 6.78,
                W == aircraft.mass*g,
                C_T >= CD, #+ (W*sing)/(0.5*rho*S*V**2),
                Vs**2  >= (2.*aircraft.mass*fs.g/rho/S/CL),
                Xgr*(2*g*(mu_b)) >= (mstall*Vs)**2,
                Xla >= Xgr,
                Sgr >= Xla,
                t >= Sgr/(0.3*V)
            ]

        return constraints, fs,perf      



class Mission(Model):
    """ Mission

    Variables
    ---------
    Srunway_to      300         [ft]        runway length
    Srunway_land    300         [ft]        runway length
    Sobstacle       400         [ft]        obstacle length
    mrunway         1.4         [-]         runway margin
    mobstacle       1.4         [-]         obstacle margin
    R               115         [nmi]       mission range
    Vstall          61          [kts]       power off stall performance
    CLstall         1.
    """
    def setup(self,poweredwheels=False,n_wheels=3,hybrid=False):
        exec parse_variables(Mission.__doc__)
        self.aircraft = Aircraft(poweredwheels,n_wheels,hybrid)
        self.takeoff = TakeOff(self.aircraft,poweredwheels,n_wheels,hybrid)
        self.obstacle_climb = Climb(self.aircraft,hybrid)
        self.climb = Climb(self.aircraft,hybrid)
        self.cruise = Cruise(self.aircraft,hybrid)
        self.landing = Landing(self.aircraft,hybrid)
        
        Wcent = Variable("W_{cent}","lbf","center aircraft weight")
        loading = self.aircraft.loading(self.cruise.flightstate,Wcent)

        self.fs = [self.takeoff,self.obstacle_climb,self.climb,self.cruise,self.landing]

        constraints = [self.obstacle_climb.h_gain == Variable("h_obstacle",50,"ft"),
                       self.climb.h_gain == Variable("h_cruise",1950,"ft"),
                       self.climb.Sclimb == Variable("Scruiseclimb",10,"miles"),
                       
                       Srunway_to >= self.takeoff.Sto*mrunway,
                       Srunway_land >= self.landing.Sgr*mrunway,
                       # Sobstacle >= Srunway*(4.0/3.0),
                       Sobstacle >= mobstacle*(self.takeoff.Sto + self.obstacle_climb.Sclimb),
                       loading.wingl["W"] == Wcent,
                       Wcent >= self.aircraft.mass*g,
                       self.obstacle_climb.perf.bw_perf.C_LC == 1.05,
                       self.climb.perf.bw_perf.C_LC == 0.95,
                       self.takeoff.perf.gen_perf.P_fuel == self.obstacle_climb.perf.gen_perf.P_fuel,
                       self.obstacle_climb.perf.gen_perf.P_fuel == self.climb.perf.gen_perf.P_fuel,
                       self.climb.perf.gen_perf.P_fuel == self.cruise.perf.gen_perf.P_fuel,
                       self.cruise.perf.gen_perf.P_fuel == self.landing.perf.gen_perf.P_fuel
        ]
        if hybrid:
            constraints += [self.aircraft.battery.E_capacity*0.8 >= self.takeoff.perf.bw_perf.P*self.takeoff.t + self.obstacle_climb.perf.bw_perf.P*self.obstacle_climb.t,
                            self.aircraft.tank.E >= sum(s.t*s.perf.gen_perf.P_fuel for s in self.fs)]
        else:
            constraints += [self.aircraft.battery.E_capacity*0.8 >= self.takeoff.E + self.obstacle_climb.E + self.climb.E + self.cruise.E]
        with gpkit.SignomialsEnabled():
            constraints += [R <= self.cruise.R]
        return constraints,self.aircraft,self.fs, loading

def writeSol(sol):
    with open('solve.txt', 'wb') as output:
        output.write(sol.table())


if __name__ == "__main__":
    poweredwheels = True
    M = Mission(poweredwheels=poweredwheels,n_wheels=3,hybrid=True)
    M.cost = M.aircraft.mass
    # M.debug()
    sol = M.localsolve("mosek")
    print sol.summary()
    writeSol(sol)

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

# CLCurves()