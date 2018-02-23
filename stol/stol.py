import os
import pandas as pd
from gpkit import Model, parse_variables
from gpkit.constraints.tight import Tight as TCS
from gpfit.fit_constraintset import FitCS
import gpkit
import math
import numpy as np
import matplotlib.pyplot as plt

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
        self.powertrain = Powertrain()
        self.battery = Battery()
        self.wing = Wing()
        self.fuselage = Fuselage()
        self.components = [self.wing,self.powertrain,self.battery,self.fuselage]
        if poweredwheels:
            self.pw = PoweredWheel()
            self.wheels = n_wheels*[self.pw]
            self.components += self.wheels
        self.mass = m
        constraints = [self.fuselage.m >= fstruct*self.mass,
                        self.mass>=sum(c["m"] for c in self.components) + Wpax*Npax]
        return constraints, self.components
    def dynamic(self,state):
        return AircraftP(self,state)


class AircraftP(Model):
    """ AircraftP

    Variables
    ---------
    L           [N]     total lift force
    D           [N]     total drag force
    LD          [-]     lift to drag ratio
    P           [kW]    total power draw
    """
    def setup(self,aircraft,state):
        exec parse_variables(AircraftP.__doc__)
        self.powertrain_perf = aircraft.powertrain.dynamic(state)
        self.wing_aero = aircraft.wing.dynamic(state)
        self.perf_models = [self.powertrain_perf,self.wing_aero]
        self.fs = state
        constraints = [L >= aircraft.mass*state["g"],
                       L <= self.wing_aero["L"],
                       P >= self.powertrain_perf["P"],
                       D >= self.wing_aero["D"] + 0.5*state.rho*aircraft.fuselage.cda*aircraft.wing.S*state.V**2,
                       LD == L/D,
                       self.powertrain_perf["T"] >= self.wing_aero["D"]
                       ]
        return constraints,self.perf_models

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
    Pstar   7   [kW/kg] motor specific power
    """
    def setup(self):
        exec parse_variables(Powertrain.__doc__)
        constraints = [m >= Pmax/Pstar]
        return constraints
    def dynamic(self,state):
        return PowertrainP(self,state)

class PowertrainP(Model):
    """ PowertrainP

    Variables
    ---------
    P               [kW]      power
    eta   0.9*0.8   [-]       whole-chain powertrain efficiency
    T               [N]       total thrust
    """
    def setup(self,powertrain,state):
        exec parse_variables(PowertrainP.__doc__)
        constraints = [T <= P*eta/state["V"],
                       P <= powertrain["Pmax"]]
        return constraints

class PoweredWheel(Model):
    """Powered Wheels

    Variables
    ---------
    RPMmax          5e3     [rpm]       maximum RPM of motor
    gear_ratio              [-]         gear ratio of powered wheel
    gear_ratio_max  10      [-]         max gear ratio of powered wheel
    tau_max                 [N*m]       torque of the
    m                       [kg]        mass of powered wheel motor
    m_ref           1       [kg]        reference mass for equations
    r               0.2     [m]         tire radius
    """
    def setup(self):
        exec parse_variables(PoweredWheel.__doc__)
        constraints = [gear_ratio <= gear_ratio_max,
                       # RPMmax == Variable("RPM_m",5500/12.3,"rpm/kg")*m,
                       tau_max <= Variable("tau_m",24.6,"N*m/kg")*m
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
    Estar       210     [Wh/kg]         specific energy
    E_capacity          [Wh]            energy capacity

    """

    def setup(self):
        exec parse_variables(Battery.__doc__)
        constraints = [m >= E_capacity/Estar]
        return constraints

class Wing(Model):
    """
    Variables
    ---------
    S               [m^2]           reference area
    b               [m]             span
    A       8       [-]             aspect ratio
    rho     6.05    [kg/m^2]        wing areal density
    m               [kg]            mass of wing
    e       0.8     [-]             span efficiency
    CLmax   3.5     [-]             max CL
    """
    def setup(self):
        exec parse_variables(Wing.__doc__)
        constraints = [m >= rho*S,
                       A == b**2/S]
        return constraints
    def dynamic(self,state):
        return WingP(self,state)

class WingP(Model):
    """

    Variables
    ---------
    L               [N]             lift force
    D               [N]             drag force
    mfac    1.1     [-]             profile drag margin factor
    CL              [-]             lift coefficient
    CD              [-]             drag coefficient
    Re              [-]             Reynolds number
    """
    def setup(self,wing,state):
        exec parse_variables(WingP.__doc__)

        constraints = [CL <= wing.CLmax,
                       CD >= mfac*1.328/Re**0.5 + CL**2/pi/wing["A"]/wing["e"],
                       L <= 0.5*CL*wing["S"]*state["rho"]*state["V"]**2,
                       D >= 0.5*CD*wing["S"]*state["rho"]*state["V"]**2,
                       Re == state["V"]*state["rho"]*(wing["S"]/wing["A"])**0.5/state["mu"]]
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
    Vstall                  [knots]     stall velocity
    zsto                    [-]         take off distance helper variable
    Sto                     [ft]        take off distance
    W                       [N]         aircraft weight
    """
    def setup(self, aircraft,poweredwheels,n_wheels):
        exec parse_variables(TakeOff.__doc__)

        fs = FlightState()

        path = os.path.dirname(os.path.abspath(__file__))
        df = pd.read_csv(path + os.sep + "logfit.csv")
        fd = df.to_dict(orient="records")[0] #fit data

        S = aircraft.wing["S"]
        Pmax = aircraft.powertrain.Pmax
        AR = aircraft.wing.A
        rho = fs.rho
        V = fs.V
        e = aircraft.wing.e
        mstall = 1.3
        constraints = [
                W == aircraft.mass*fs.g,
                T/W >= A/g + mu,
                B >= g/W*0.5*rho*S*CDg,
                T <= Pmax*0.8*0.9/V,
                CDg >= cda + cdp + aircraft.wing.CLmax**2/pi/AR/e,
                Vstall == (2*W/rho/S/aircraft.wing.CLmax)**0.5,
                V == mstall*Vstall,
                FitCS(fd, zsto, [A/g, B*V**2/g]), #fit constraint set, pass in fit data, zsto is the 
                # y variable, then arr of independent (input) vars, watch the units
                Sto >= 1.0/2.0/B*zsto]
        if poweredwheels:
            wheel_models = [wheel.dynamic(fs) for wheel in aircraft.wheels]
            with gpkit.SignomialsEnabled():
                constraints += [T <= Pmax*0.8*0.9/V + sum(model.T for model in wheel_models)]
                constraints += wheel_models
        else:
            constraints += [T <= Pmax*0.8*0.9/V]

        return constraints, fs

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
    """

    def setup(self,aircraft):
        exec parse_variables(Climb.__doc__)
        self.flightstate = FlightState()
        perf = aircraft.dynamic(self.flightstate)

        CL = self.CL = perf.wing_aero.CL
        S = self.S = aircraft.wing.S
        CD = self.CD = perf.wing_aero.CD
        V = perf.fs.V
        rho = perf.fs.rho

        constraints = [
            W ==  aircraft.mass*perf.fs.g,
            W <= 0.5*CL*rho*S*V**2,
            perf.powertrain_perf.T >= perf.D + W*h_dot/V,
            h_gain <= h_dot*t,
            Sclimb == V*t, #sketchy constraint, is wrong with cos(climb angle)
            perf.P >= perf.powertrain_perf.T*V/0.8,
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
        constraints = [R <= aircraft.battery["Estar"]*self.perf.L/self.perf.D * (aircraft.battery["m"]/aircraft.mass) * self.perf.powertrain_perf.eta * 1/self.flightstate["g"],
                       E >= self.perf.topvar("P")*t,
                       t >= R/self.flightstate["V"],
                       self.flightstate["V"] >= Vmin]
        return constraints, self.perf
        
class GLanding(Model):
    """ Glanding model

    Variables
    ---------
    g           9.81        [m/s**2]    gravitational constant
    gload       0.5         [-]         gloading constant
    Vstall                  [knots]     stall velocity
    Sgr                     [ft]        landing ground roll
    msafety     1.4         [-]         Landing safety margin
    """
    def setup(self, aircraft):
        exec parse_variables(GLanding.__doc__)

        fs = FlightState()

        S = self.S = aircraft.wing.S
        rho = fs.rho
        V = fs.V
        mstall = 1.3

        constraints = [
            Sgr >= 0.5*V**2/gload/g,
            Vstall == (2.*aircraft.mass*fs.g/rho/S/aircraft.wing.CLmax)**0.5,
            V >= mstall*Vstall,
            ]

        return constraints, fs

class Mission(Model):
    """ Mission

    Variables
    ---------
    Srunway     300         [ft]        runway length
    Sobstacle   400         [ft]        obstacle length
    mrunway     1.4         [-]         runway margin
    R           115         [nmi]       mission range
    """
    def setup(self,poweredwheels=False,n_wheels=3):
        exec parse_variables(Mission.__doc__)
        self.aircraft = Aircraft(poweredwheels,n_wheels)
        takeoff = TakeOff(self.aircraft,poweredwheels,n_wheels)
        obstacle_climb = Climb(self.aircraft)
        self.fs = [takeoff,obstacle_climb,Cruise(self.aircraft),GLanding(self.aircraft)]
        constraints = [Srunway >= self.fs[0].Sto*mrunway,
                       Sobstacle >= self.fs[0].Sto + self.fs[1].Sclimb,
                       self.fs[3].Sgr*mrunway <= Srunway,
                       R <= self.fs[2]["R"],
                       self.aircraft.battery.E_capacity >= self.fs[1].E + self.fs[2].E]
        return constraints,self.aircraft,self.fs

# if __name__ == "__main__":
#     poweredwheels = False
#     M = Mission(poweredwheels=poweredwheels,n_wheels=3)
#     # M.substitutions.update({M.Srunway:('sweep', np.linspace(40,300,10))})
#     M.cost = M.aircraft.mass
#     M.debug()
#     if poweredwheels == True:
#       sol = M.localsolve("mosek")
#     else:
#       sol = M.solve("mosek")
#     print sol.table()
#     print sol(M.Srunway)
#     print sol(M.aircraft.mass)
#     plt.plot(sol(M.aircraft.wing.b),sol(M.Srunway))
    # plt.show()

def CLCurves():
    M = Mission()
    runway_sweep = np.linspace(100,300,20)
    runway_factor = 3
    M.substitutions.update({M.Srunway:('sweep',np.linspace(100,300,20))})
    M.substitutions.update()
    M.cost = M.aircraft.mass
    CLmax_set = np.linspace(3.5,8,5)
    for CLmax in CLmax_set:
        print CLmax
        M.substitutions.update({M.aircraft.wing.CLmax:CLmax})
        sol = M.solve("mosek")
        print sol(M.aircraft.mass)
        plt.plot(sol(M.Srunway),sol(M.aircraft.mass),label="$CL_{max} = $" + str(CLmax))
    
    plt.grid()
    # plt.xlim([0,300])
    # plt.ylim([0,1600])
    plt.title("Runway length requirement for eSTOL")
    plt.xlabel("Runway length [ft]")
    plt.ylabel("Aircraft mass [kg]")
    plt.legend()
    plt.show()

CLCurves()