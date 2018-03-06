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
from gpkitmodels.GP.aircraft.wing.boxspar import BoxSpar
from gpkitmodels.GP.aircraft.wing.wing_skin import WingSkin
from gpkitmodels.GP.aircraft.wing.wing import Planform
from decimal import *
pi = math.pi
class Aircraft(Model):
    """ Aircraft

    Variables
    ---------
    m           [kg]    aircraft mass
    Npax    4   [-]     number of passengers
    Wpax    90  [kg]    mass of a passenger
    fstruct 0.3 [-]     structural mass fraction
    """
    def setup(self,poweredwheels=False,n_wheels=3):
        exec parse_variables(Aircraft.__doc__)
        # self.powertrain = Powertrain()
        self.battery = Battery()
        # self.wing = Wing()
        self.fuselage = Fuselage()
        self.bw = BlownWing()
        self.components = [self.bw,self.battery,self.fuselage]
        if poweredwheels:
            self.pw = PoweredWheel()
            self.wheels = n_wheels*[self.pw]
            self.components += self.wheels
        self.mass = m
        constraints = [self.fuselage.m >= fstruct*self.mass,
                        self.mass>=sum(c.topvar("m") for c in self.components) + Wpax*Npax]
        return constraints, self.components
    def dynamic(self,state):
        return AircraftP(self,state)
    def loading(self,Wcent,state):
        return AircraftLoading(self,Wcent,state)

class AircraftP(Model):
    """ AircraftP

    Variables
    ---------
    P           [kW]    total power draw
    """
    def setup(self,aircraft,state):
        exec parse_variables(AircraftP.__doc__)
        self.bw_perf = aircraft.bw.dynamic(state)
        self.batt_perf = aircraft.battery.dynamic(state)
        self.perf_models = [self.bw_perf,self.batt_perf]
        self.fs = state
        constraints = [0.5*self.bw_perf.C_L*state.rho*aircraft.bw.wing["S"]*state.V**2 >= aircraft.mass*state["g"],
                       P == self.bw_perf["P"],
                       P <= aircraft.bw.powertrain["Pmax"],
                       self.batt_perf.P >= P,
                       self.bw_perf.C_T >= self.bw_perf.C_D + aircraft.fuselage.cda,
                       # D >= self.wing_aero["D"] + 0.5*state.rho*aircraft.fuselage.cda*aircraft.wing.S*state.V**2,
                       # self.powertrain_perf["T"] >= self.wing_aero["D"]
                       ]
        return constraints,self.perf_models

class AircraftLoading(Model):
    def setup(self,aircraft,state,Wcent):
        self.wingl = aircraft.bw.wing.spar.loading(aircraft.bw.wing)
        loading = [self.wingl]
        return loading
class Fuselage(Model):
    """ Fuselage

    Variables
    ---------
    m               [kg]    mass of fuselage
    cda     0.015   [-]     parasite drag coefficient
    """
    def setup(self):
        exec parse_variables(Fuselage.__doc__)

#   def dynamic(self,state):
#       return FuselageP(self,state)

# class FuselageP(Model):
#   """ FuselageP

#   Variables
#   ---------
#   D       [N]     total drag force from fuselage
#   """
#   def setup(self,state):
#       exec parse_variables(FuselageP.__doc__)

class Powertrain(Model):
    """ Powertrain
    Variables
    ---------
    m           [kg]    powertrain mass
    Pmax        [kW]    maximum power
    Pstar   5   [kW/kg] motor specific power
    r     0.2   [m]     propeller radius
    """
    def setup(self):
        exec parse_variables(Powertrain.__doc__)
        constraints = [m >= Pmax/Pstar]
        return constraints

# class PowertrainP(Model):
#     """ PowertrainP

#     Variables
#     ---------
#     P               [kW]      power
#     eta   0.9*0.8   [-]       whole-chain powertrain efficiency
#     T               [N]       total thrust
#     """
#     def setup(self,powertrain,state):
#         exec parse_variables(PowertrainP.__doc__)
#         constraints = [T <= P*eta/state["V"],
#                        P <= powertrain["Pmax"]]
#         return constraints

class PoweredWheel(Model):
    """Powered Wheels

    Variables
    ---------
    RPMmax          4e3     [rpm]       maximum RPM of motor
    gear_ratio              [-]         gear ratio of powered wheel
    gear_ratio_max  10      [-]         max gear ratio of powered wheel
    tau_max                 [N*m]       torque of the
    m                       [kg]        mass of powered wheel motor
    m_ref           1       [kg]        reference mass for equations
    r               0.2     [m]         tire radius
    """
    def setup(self):
        exec parse_variables(PoweredWheel.__doc__)
        #Currently using the worst values
        constraints = [gear_ratio <= gear_ratio_max,
                       # RPMmax == Variable("RPM_m",5500/12.3,"rpm/kg")*m,
                       tau_max <= Variable("tau_m",14.3,"N*m/kg")*m
                       # 27.3*m >= Variable("tau_m",1,"kg/(N*m)")*tau_max + 80.2*m_ref
                       ]
        return constraints
    def dynamic(self,state):
        return PoweredWheelP(self,state)

class PoweredWheelP(Model):
    """ PoweredWheelsP
    Variables
    ---------
    RPM             [rpm]       rpm of powered wheel
    tau             [N*m]       torque of powered wheel
    T               [N]         thrust from powered wheel
    """
    def setup(self,pw,state):
        exec parse_variables(PoweredWheelP.__doc__)
        constraints =[state.V <= pw.RPMmax*2*pi*pw.r/pw.gear_ratio,
                      state.V == RPM*2*pi*pw.r/pw.gear_ratio,
                      T == tau*pw.gear_ratio/pw.r,
                      tau <= pw.tau_max]
        return constraints
class Battery(Model):
    """ Battery
    Variables
    ---------
    m                   [kg]            total mass
    Estar       150     [Wh/kg]         specific energy
    E_capacity          [Wh]            energy capacity
    """

    def setup(self):
        exec parse_variables(Battery.__doc__)
        constraints = [m >= E_capacity/Estar]
        return constraints
    def dynamic(self,state):
        return BatteryP(self,state)

class BatteryP(Model):
    """BatteryP
    Variables
    ---------
    P                   [kW]        battery power draw
    Pstar      2        [kW/kg]     battery specific power limit
    """
    def setup(self,batt,state):
        exec parse_variables(BatteryP.__doc__)
        constraints = [P <=  Pstar*batt.m]
        return constraints

class Wing(Model):
    """
    Wing Model
    Variables
    ---------
    W                   [lbf]       wing weight
    mfac        1.2     [-]         wing weight margin factor
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

    sparModel = BoxSpar
    fillModel = False
    skinModel = WingSkin

    def setup(self, N=5):
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

        constraints = [W/mfac >= sum(c["W"] for c in self.components)]

        return constraints, self.planform, self.components

class BlownWing(Model):
    """
    Variables
    ---------
    n_prop    8     [-]             number of props
    m               [kg]            mass
    """
    def setup(self):
        exec parse_variables(BlownWing.__doc__)
        self.powertrain = Powertrain()
        N = 14
        self.wing = Wing(N)
        self.wing.substitutions[self.wing.planform.tau]=0.12

        constraints = [
        m >= self.powertrain["m"] + self.wing.topvar("W")/Variable("g",9.8,"m/s/s")
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
    C_LC    2.5     [-]             lift coefficient due to circulation
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
    J_prime         [kg/(s^2)]      momentum flow per unit span
    E_prime         [J/(m*s)]       energy flow per unit span
    rho_j   1.225   [kg/m^3]        density in jet flow
    u_j             [m/s]           velocity in jet flow
    h               [m]             Wake height
    T               [N]             propeller thrust
    P               [kW]            power draw
    A_disk          [m^2]           area of prop disk

    """
    def setup(self,bw,state):
        #bw is a BlownWing object
        #state is a FlightState
        exec parse_variables(BlownWingP.__doc__)
        with gpkit.SignomialsEnabled():
            constraints = [
            A_disk == pi*bw.powertrain.r**2,
            (P/(0.5*T*state["V"]) - 1)**2 >= (T/(A_disk*(state.V**2)*state.rho/2)+1),
            (u_j/state.V)**2 <= (T/(A_disk*(state.V**2)*state.rho/2) + 1),

            P <= bw.powertrain["Pmax"],
            C_L <= C_LC*(1+2*C_J/(pi*bw.wing["AR"]*e)),
            
            C_T == T/((0.5*state.rho*state.V**2)*bw.wing["S"]),
            m_dotprime == rho_j*u_j*h,
            J_prime ==  m_dotprime*u_j,
            E_prime == 0.5*m_dotprime*u_j**2,
            C_Q ==  m_dotprime/(state.rho*state.V* bw.wing["cmac"]),
            C_J == J_prime/(0.5*state.rho*state.V**2 * bw.wing["cmac"]),
            C_E == E_prime/(0.5*state.rho*state.V**3 * bw.wing["cmac"]),
            h == (bw.n_prop*pi*bw.powertrain.r**2)/bw.wing.b,
            C_Di == (C_L**2)/(pi*bw.wing["AR"]*e),
            C_D >= C_Di  + C_Dp,
            C_Dp == mfac*1.328/Re**0.5, #friction drag only, need to add form
            Re == state["V"]*state["rho"]*(bw.wing["S"]/bw.wing["AR"])**0.5/state["mu"],
            # C_T >= C_D #steady level non-accelerating constraint as C_T-C_D = 1
            ]
        return constraints
class FlightState(Model):
    """ Flight State

    Variables
    ---------
    rho         1.225       [kg/m^3]        air density
    mu          1.789e-5    [kg/m/s]        air viscosity
    V                       [knots]         speed
    g           9.8         [m/s/s]         acceleration due to gravity
    """
    def setup(self):
        exec parse_variables(FlightState.__doc__)

class TakeOff(Model):
    """
    take off model
    http://www.dept.aoe.vt.edu/~lutze/AOE3104/takeoff&landing.pdf

    Variables
    ---------
    A                       [m/s**2]    log fit equation helper 1
    B                       [1/m]       log fit equation helper 2
    g           9.81        [m/s**2]    gravitational constant
    mu          0.025       [-]         coefficient of friction
    T                       [N]       take off thrust
    cda         0.015       [-]         parasite drag coefficient
    CDg                     [-]         drag ground coefficient
    cdp         0.025       [-]         profile drag at Vstallx1.2
    Kg          0.04        [-]         ground-effect induced drag parameter
    zsto                    [-]         take off distance helper variable
    Sto                     [ft]        take off distance
    W                       [N]         aircraft weight
    E                       [kWh]       energy consumed in takeoff
    """
    def setup(self, aircraft,poweredwheels,n_wheels):
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
        perf = aircraft.dynamic(fs)
        e = perf.bw_perf.e
        mstall = 1.3
        constraints = [
                W == aircraft.mass*fs.g,
                T/W >= A/g + mu,
                B >= g/W*0.5*rho*S*CDg,
                CDg >= perf.bw_perf.C_D,
                V >= mstall*(2*W/rho/S/perf.bw_perf.C_L)**0.5,
                FitCS(fd, zsto, [A/g, B*V**2/g]), #fit constraint set, pass in fit data, zsto is the 
                # y variable, then arr of independent (input) vars, watch the units
                E >= (Sto/V)*perf.bw_perf.P,
                Sto >= 1.0/2.0/B*zsto]
        if poweredwheels:
            wheel_models = [wheel.dynamic(fs) for wheel in aircraft.wheels]
            with gpkit.SignomialsEnabled():
                constraints += [T <= perf.bw_perf.T + sum(model.T for model in wheel_models)]
                constraints += wheel_models

        else:
            constraints += [T <= perf.bw_perf.T]

        return constraints, fs, perf

class Climb(Model):

    """ Climb model

    Variables
    ---------
    Sclimb                  [ft]        distance covered in climb
    h_gain        50        [ft]        height gained in climb
    t                       [s]         time of climb
    h_dot                   [m/s]       climb rate
    E                       [kWh]       climb energy usage
    W                       [N]         aircraft weight
    
    LaTex Strings
    -------------
    Sclimb      S_{\\mathrm{climb}}
    h_gain      h_{\\mathrm{gain}}
    h_dot       \dot{h}
    """

    def setup(self,aircraft):
        exec parse_variables(Climb.__doc__)
        self.flightstate = FlightState()
        perf = aircraft.dynamic(self.flightstate)

        CL = self.CL = perf.bw_perf.C_L
        S = self.S = aircraft.bw.wing["S"]
        CD = self.CD = perf.bw_perf.C_D
        V = perf.fs.V
        rho = perf.fs.rho

        constraints = [
            W ==  aircraft.mass*perf.fs.g,
            W <= 0.5*CL*rho*S*V**2,
            perf.bw_perf.T >= 0.5*CD*rho*S*V**2 + W*h_dot/V,
            h_gain <= h_dot*t,
            Sclimb == V*t, #sketchy constraint, is wrong with cos(climb angle)
            perf.P >= perf.bw_perf.T*V/0.8,
            E >= perf.P*t
        ]
        return constraints, perf

class Cruise(Model):
    """

    Variables
    ---------
    E           [kWh]       Energy consumed in flight segment
    R           [nmi]       Range flown in flight segment
    t           [min]       Time to fly flight segment
    Vmin  120   [kts]       Minimum flight speed
    """

    def setup(self,aircraft):
        exec parse_variables(Cruise.__doc__)
        self.flightstate = FlightState()
        self.perf = aircraft.dynamic(self.flightstate)
        constraints = [R <= t*self.flightstate.V,
                       E >= self.perf.topvar("P")*t,
                       self.flightstate["V"] >= Vmin]
        return constraints, self.perf
        
class GLanding(Model):
    """ Glanding model

    Variables
    ---------
    g           9.81        [m/s**2]    gravitational constant
    gload       0.5         [-]         gloading constant
    Sgr                     [ft]        landing ground roll
    msafety     1.4         [-]         Landing safety margin
    t                       [s]         Time to execute landing
    E                       [kWh]       Energy used in landing
    """
    def setup(self, aircraft):
        exec parse_variables(GLanding.__doc__)

        fs = FlightState()

        S = self.S = aircraft.bw.wing["S"]
        rho = fs.rho
        mstall = 1.3
        perf = aircraft.dynamic(fs)
        constraints = [
            Sgr >= 0.5*fs.V**2/gload/g,
            fs.V >= mstall*(2.*aircraft.mass*fs.g/rho/S/perf.bw_perf.C_L)**0.5,
            t >= Sgr/fs.V,
            E >= t*perf.bw_perf.P
        ]

        return constraints, fs,perf

class Mission(Model):
    """ Mission

    Variables
    ---------
    Srunway     300         [ft]        runway length
    Sobstacle               [ft]        obstacle length
    mrunway     1.4         [-]         runway margin
    mobstacle   1.4         [-]         obstacle margin
    R           115         [nmi]       mission range
    """
    def setup(self,poweredwheels=False,n_wheels=3):
        exec parse_variables(Mission.__doc__)
        self.aircraft = Aircraft(poweredwheels,n_wheels)
        takeoff = TakeOff(self.aircraft,poweredwheels,n_wheels)
        climb = Climb(self.aircraft)
        cruise = Cruise(self.aircraft)
        landing = GLanding(self.aircraft)
        
        Wcent = Variable("W_{cent}","lbf","center aircraft weight")
        loading = self.aircraft.loading(cruise.flightstate,Wcent)

        self.fs = [takeoff,climb,cruise,landing]
        constraints = [Srunway >= takeoff.Sto*mrunway,
                       Srunway >= landing.Sgr*mrunway,
                       Sobstacle == Srunway*(4/3),
                       Sobstacle >= mobstacle*(takeoff.Sto + climb.Sclimb),
                       R <= cruise.R,
                       self.aircraft.battery.E_capacity >= takeoff.E + climb.E + cruise.E + landing.E,
                       Wcent >= self.aircraft.mass*g,
                       Wcent == loading.wingl["W"]
        ]
        return constraints,self.aircraft,self.fs, loading

if __name__ == "__main__":
    poweredwheels = True
    M = Mission(poweredwheels=poweredwheels,n_wheels=3)
    M.cost = M.aircraft.mass
    # M.debug()
    sol = M.localsolve("mosek")
    print sol.summary()

# def CLCurves():
#     M = Mission(poweredwheels=True,n_wheels=3)
#     runway_sweep = np.linspace(100,300,4)
#     obstacle_sweep = runway_sweep*4/3
#     M.substitutions.update({M.Srunway:('sweep',runway_sweep)})
#     M.cost = M.aircraft.mass
#     sol = M.localsolve("mosek")
#     print sol.summary()
#     plt.plot(sol(M.Srunway),sol(M.aircraft.mass))

#     # sol = M.solve()
#     # # CLmax_set = np.linspace(3.5,8,5)
#     # for CLmax in CLmax_set:
#     #     print CLmax
#     #     # M.substitutions.update({M.aircraft.bw.CLmax:CLmax})
#     #     sol = M.solve("mosek")
#     #     print sol(M.aircraft.mass)
    
#     plt.grid()
#     # plt.xlim([0,300])
#     # plt.ylim([0,1600])
#     plt.title("Runway length requirement for eSTOL")
#     plt.xlabel("Runway length [ft]")
#     plt.ylabel("Takeoff mass [kg]")
#     # plt.legend()
#     plt.show()

# CLCurves()