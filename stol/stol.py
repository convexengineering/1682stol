" short take off and landing aircraft model "
import os
import pandas as pd
from numpy import pi
from gpkit import Model, parse_variables
from gpkit.constraints.tight import Tight as TCS
from gpfit.fit_constraintset import FitCS


class FlightState(Model):
    """ Flight State

    Variables
    ---------
    rho         1.225       [kg/m^3]        air density
    mu          1.789e-5    [kg/m/s]        air viscosity
    V                       [knots]         speed
    """
    def setup(self):
        exec parse_variables(FlightState.__doc__)

class AircraftPerf(Model):
    """ Simple Drag model

    Variables
    ---------
    CD              [-]         drag coefficient
    Re              [-]         Reynolds number
    Cf              [-]         coefficient of friction
    CL              [-]         coefficient of lift
    mfac    1.1     [-]         profile drag margin factor
    T               [N]         thrust
    Pshaft          [W]         shaft power
    etaprop 0.8     [-]         prop efficiency
    """
    def setup(self, aircraft):
        exec parse_variables(AircraftPerf.__doc__)

        self.fs = FlightState()

        rho = self.fs.rho
        V = self.fs.V
        S = aircraft.S
        AR = aircraft.AR
        mu = self.fs.mu
        cda = aircraft.cda
        e = aircraft.e

        constraints = [CD >= cda + 2.*Cf*mfac + CL**2/pi/AR/e,
                       Re == V*rho*(S/AR)**0.5/mu,
                       Cf >= 1.328/Re**0.5]

        return constraints, self.fs

class Aircraft(Model):
    """ thing that we want to build

    Variables
    ---------
    W                           [lbf]           aircraft weight
    WS                          [lbf/ft^2]      Aircraft wing loading
    PW                          [hp/lbf]        Aircraft shaft hp/weight ratio
    Npax            4           [-]             number of seats
    Wpax            195         [lbf]           passenger weight
    hbatt           210         [W*hr/kg]       battery specific energy
    etae            0.9         [-]             total electrical efficiency
    Wbatt                       [lbf]           battery weight
    Wwing                       [lbf]           wing weight
    Pshaftmax                   [W]             max shaft power
    sp_motor        7./9.81     [kW/N]          Motor specific power
    Wmotor                      [lbf]           motor weight
    fstruct         0.3         [-]             structural weight fraction
    Wstruct                     [lbf]           structural weight
    S                           [ft^2]   `       wing area
    AR              8           [-]             aspect ratio
    WS2             15./24      [lbf/ft^2]      wing weight per area
    cda             0.015       [-]             parasite drag coefficient
    e               0.8         [-]             span efficiency
    mstall          1.3         [-]             stall margin
    """

    flight_model = AircraftPerf

    def setup(self):
        exec parse_variables(Aircraft.__doc__)

        return [
            WS == W/S,
            PW == Pshaftmax/W,
            TCS([W >= Wbatt + Wpax*Npax+ WS2*S + Wmotor + Wstruct]),
            Wstruct >= fstruct*W,
            Wmotor >= Pshaftmax/sp_motor,
            ]

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
    T                       [lbf]       take off thrust
    cda         0.024       [-]         parasite drag coefficient
    CDg                     [-]         drag ground coefficient
    cdp         0.025       [-]         profile drag at Vstallx1.2
    Kg          0.04        [-]         ground-effect induced drag parameter
    CLto        3.5         [-]         max lift coefficient
    Vstall                  [knots]     stall velocity
    zsto                    [-]         take off distance helper variable
    Sto                     [ft]        take off distance
    """
    def setup(self, aircraft):
        exec parse_variables(TakeOff.__doc__)

        fs = FlightState()

        path = os.path.dirname(os.path.abspath(__file__))
        df = pd.read_csv(path + os.sep + "logfit.csv")
        fd = df.to_dict(orient="records")[0]

        S = self.S = aircraft.S
        W = self.W = aircraft.W
        Pshaftmax = aircraft.Pshaftmax
        AR = aircraft.AR
        rho = fs.rho
        V = fs.V
        cda = aircraft.cda
        e = aircraft.e
        mstall = aircraft.mstall

        constraints = [
            T/W >= A/g + mu,
            B >= g/W*0.5*rho*S*CDg,
            T <= Pshaftmax*0.8/V,
            CDg >= cda + cdp + CLto**2/pi/AR/e,
            Vstall == (2*W/rho/S/CLto)**0.5,
            V == mstall*Vstall,
            FitCS(fd, zsto, [A/g, B*V**2/g]),
            Sto >= 1.0/2.0/B*zsto]

        return constraints, fs

class Climb(Model):

    """ Climb model

    Variables
    ---------
    Sclimb                  [ft]        distance covered in climb
    h_gain     50           [ft]        height gained in climb
    t                       [s]         time of climb
    h_dot                   [m/s]       climb rate
    E                       [kWh]       climb energy usage
    """

    def setup(self,aircraft):
        exec parse_variables(Climb.__doc__)
        perf = aircraft.flight_model(aircraft)

        CL = self.CL = perf.CL
        S = self.S = aircraft.S
        CD = self.CD = perf.CD
        W = self.W = aircraft.W
        V = perf.fs.V
        rho = perf.fs.rho

        constraints = [
            W <= 0.5*CL*rho*S*V**2,
            perf.T >= 0.5*CD*rho*S*V**2 + W*h_dot/V,
            h_gain <= h_dot*t,
            Sclimb == V*t, #sketchy constraint, is wrong with cos(climb angle)
            perf.Pshaft >= perf.T*V/perf.etaprop,
            E >= perf.Pshaft*t
        ]
        return constraints, perf

class Cruise(Model):
    """ Aicraft Range model

    Variables
    ---------
    R               100    [nmi]     aircraft range
    g               9.81   [m/s**2]  gravitational constant
    treserve        30.0   [min]     Reserve flight time
    Vmin            120    [kts]     min speed
    t                      [min]     flight time
    """
    def setup(self, aircraft):
        exec parse_variables(Cruise.__doc__)

        perf = aircraft.flight_model(aircraft)

        CL = self.CL = perf.CL
        S = self.S = aircraft.S
        CD = self.CD = perf.CD
        W = self.W = aircraft.W
        hbatt = aircraft.hbatt
        Wbatt = aircraft.Wbatt
        etae = aircraft.etae
        V = perf.fs.V
        rho = perf.fs.rho
        self.Pshaft = perf.Pshaft
        constraints = [
            W == 0.5*CL*rho*S*V**2,
            perf.T >= 0.5*CD*rho*S*V**2,
            perf.Pshaft >= perf.T*V/perf.etaprop,
            V >= Vmin,
            R + treserve*V <= hbatt*Wbatt/g*etae*V/perf.Pshaft,
            t >= R/V]

        return constraints, perf

class GLanding(Model):
    """ Glanding model

    Variables
    ---------
    g           9.81        [m/s**2]    gravitational constant
    gload       0.5         [-]         gloading constant
    Vstall                  [knots]     stall velocity
    Sgr                     [ft]        landing ground roll
    msafety     1.4         [-]         Landing safety margin
    CLland      3.5         [-]         landing CL
    """
    def setup(self, aircraft):
        exec parse_variables(GLanding.__doc__)

        fs = FlightState()

        S = self.S = aircraft.S
        W = self.W = aircraft.W
        rho = fs.rho
        V = fs.Vprofbit
        mstall = aircraft.mstall

        constraints = [
            Sgr >= 0.5*V**2/gload/g,
            Vstall == (2.*W/rho/S/CLland)**0.5,
            V >= mstall*Vstall,
            ]

        return constraints, fs

class Mission(Model):
    """ Mission

    Variables
    ---------
    Srunway     300     [ft]        runway length
    Sobstacle   400     [ft]        obstacle length
    mrunway     1.4     [-]         runway margin
    g           9.81    [m/s**2]    gravitational constant
    """
    def setup(self):
        exec parse_variables(Mission.__doc__)

        self.aircraft = Aircraft()

        self.takeoff = TakeOff(self.aircraft)
        self.obstacle_climb = Climb(self.aircraft)
        self.cruise = Cruise(self.aircraft)
        self.landing = GLanding(self.aircraft)
        self.mission = [self.takeoff, self.obstacle_climb, self.cruise, self.landing]

        Pshaftmax = self.aircraft.Pshaftmax
        Pshaft = self.cruise.Pshaft
        Sto = self.takeoff.Sto
        Slnd = self.landing.Sgr
        Sclimb = self.obstacle_climb["Sclimb"]

        constraints = [Pshaftmax >= Pshaft,
                       Srunway >= mrunway*Sto,
                       Srunway >= mrunway*Slnd,
                       Sobstacle >= Sto + Sclimb,
                       self.aircraft.Wbatt*self.aircraft.hbatt/g >= self.obstacle_climb["E"] + self.cruise["t"]*self.cruise.Pshaft,
                    ]

        return constraints, self.aircraft, self.mission


if __name__ == "__main__":
    M = Mission()
    M.substitutions.update({M.cruise.R: 100, M.Srunway: 400})
    M.cost = M[M.aircraft.W]
    sol = M.solve("mosek")
    print sol.summary()
