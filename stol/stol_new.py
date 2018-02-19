from gpkit import Model, Variable

class Aircraft(Model):
	""" Aircraft

	Variables
	---------
	m 			[kg] 	aircraft mass
	"""
	def setup(self):
		exec parse_variable(Aircraft.__doc__)
	def dynamic(self,state):
		return AircraftP(self,state)

class AircraftP(Model):
	""" AircraftP

	Variables
	---------
	L 			[N]		total lift force
	D 			[N]		total drag force
	"""
	def setup(self,aircraft):
		exec parse_variables(AircraftP.__doc__)
		self.powertrain_perf = aircraft

class Wing(Model):
	"""
	Variables
	---------
	S 			[m^2]			reference area
	b			[m]				span
	A 			[-]				aspect ratio
	rho			[lbf/ft^2]		wing areal density
	"""
	def setup(self):
		exec parse_variables(Wing.__doc__)


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

class Cruise(Model):
	"""

	Variables
	---------
	E 		[kWh]		Energy consumed in flight segment
	R 		[nmi]		Range flown in flight segment
	t		[min]		Time to fly flight segment
	"""

	def setup(self,aircraft):
		exec parse_variables(FlightState.__doc__)
		self.flighstate = FlightState()
		self.perf = aircraft.dynamic(self.flightstate)
		constraints = [R == self.perf.L/self.perf.D * (aircraft.battery.mass/aircraft.mass) * self.perf.powertrain_perf.eta * 1/self.flightstate["g"]]
		return constraints, perf
		

class Mission(Model):
	""" Mission

	Variables
	---------
    Srunway     300     [ft]        runway length
    Sobstacle   400     [ft]        obstacle length
    mrunway     1.4     [-]         runway margin
   	"""
	def setup(self,R):
		exec parse_variables(Mission.__doc__)
		self.fs = [Cruise()]
		R = sum(comp.topvar("R") for fs in self.fs)

if __name__ == "__main__":
    M = Mission(Variable("range_req",100,"nmi"))
    M.cost = 1/M["R"]
    sol = M.solve("mosek")
    print sol.summary()
