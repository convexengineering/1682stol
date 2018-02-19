from gpkit import Model, Variable,parse_variables
import math
pi = math.pi
class Aircraft(Model):
	""" Aircraft

	Variables
	---------
	m 			[kg] 	aircraft mass
	"""
	def setup(self):
		exec parse_variables(Aircraft.__doc__)
		self.powertrain = Powertrain()
		self.battery = Battery()
		self.wing = Wing()
		self.components = [self.wing,self.powertrain,self.battery]
		self.mass = m
		constraints = [m>=sum(c["m"] for c in self.components)]
		return constraints, self.components
	def dynamic(self,state):
		return AircraftP(self,state)


class AircraftP(Model):
	""" AircraftP

	Variables
	---------
	L 			[N]		total lift force
	D 			[N]		total drag force
	P 			[kW] 	total power draw
	"""
	def setup(self,aircraft,state):
		exec parse_variables(AircraftP.__doc__)
		self.powertrain_perf = aircraft.powertrain.dynamic(state)
		self.wing_aero = aircraft.wing.dynamic(state)
		self.perf_models = [self.powertrain_perf,self.wing_aero]

		constraints = [L >= aircraft.mass*state["g"],
					   L <= self.wing_aero["L"],
					   P >= self.powertrain_perf["P"],
					   D >= self.wing_aero["D"],
					   self.powertrain_perf["T"] >= self.wing_aero["D"]
					   ]
		return constraints,self.perf_models

class Powertrain(Model):
	""" Powertrain
	Variables
	---------
	m 			[kg]	powertrain mass
	Pmax		[kW]	maximum power
	Pstar	7	[kW/kg]	motor specific power
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
	P 			[kW] 	  power
	eta   0.9	[-]		  whole-chain powertrain efficiency
	T 			[N]		  total thrust
	"""
	def setup(self,powertrain,state):
		exec parse_variables(PowertrainP.__doc__)
		constraints = [T <= P*eta/state["V"],
					   P <= powertrain["Pmax"]]
		return constraints

class Battery(Model):
	""" Battery
	Variables
	---------
	m 					[kg]   			total mass
	Estar  		210		[Wh/kg]			specific energy
	E_capacity  		[Wh]			energy capacity

	"""

	def setup(self):
		exec parse_variables(Battery.__doc__)
		constraints = [m >= E_capacity/Estar]
		return constraints

class Wing(Model):
	"""
	Variables
	---------
	S 	2		[m^2]			reference area
	b			[m]				span
	A 	8		[-]				aspect ratio
	rho	73.28	[kg/m^2]		wing areal density
	m 			[kg]			mass of wing
	e 	0.8		[-]				span efficiency
	"""
	def setup(self):
		exec parse_variables(Wing.__doc__)
		constraints = [m >= rho*S]
		return constraints
	def dynamic(self,state):
		return WingP(self,state)

class WingP(Model):
	"""

	Variables
	---------
	L 				[N]				lift force
	D           	[N]				drag force
	CLmax	3.5 	[-]				max CL
	mfac	1.1 	[-]				profile drag margin factor
	CL 				[-]				lift coefficient
	CD 				[-] 			drag coefficient
	Re 				[-]				Reynolds number
	"""
	def setup(self,wing,state):
		exec parse_variables(WingP.__doc__)

		constraints = [CL <= CLmax,
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
    g 			9.8			[m/s/s]			acceleration due to gravity
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
	"""

	Variables
	---------
	E 			[kWh]		Energy consumed in flight segment
	R 			[nmi]		Range flown in flight segment
	t			[min]		Time to fly flight segment
	Vmin  120	[kts]		Minimum flight speed
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
    R 			120		[nmi]		mission range
   	"""
	def setup(self):
		exec parse_variables(Mission.__doc__)
		self.aircraft = Aircraft()
		self.fs = [Cruise(self.aircraft)]		
		constraints = [R <= self.fs[0]["R"]]
		constraints += [self.aircraft.battery.E_capacity >= sum(fs["E"] for fs in self.fs)]
		return constraints,self.aircraft,self.fs

if __name__ == "__main__":
    M = Mission()
    M.cost = 1/M.topvar("R")
    M.debug()
    sol = M.solve("mosek")
    print sol.summary()
