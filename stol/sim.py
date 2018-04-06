from gpkit import Model, Variable

class Aircraft(Model):
    """ Aircraft

    Variables
    ---------
    m                   [kg]    aircraft mass
    """
    def setup(self,poweredwheels=False,n_wheels=3,hybrid=False):
        exec parse_variables(Aircraft.__doc__)
        return constraints
    def dynamic(self,state,hybrid=False,powermode="batt-chrg",t_charge=None):
        return AircraftP(self,state,hybrid,powermode=powermode,t_charge=t_charge)

class Mission(Model):
    """ Mission

    Variables
    ---------
    score	3	[m/s * ft]	performance score for vehicle
    """
    def setup(self):
        exec parse_variables(Mission.__doc__)

if __name__ == "__main__":
    M = Mission()
    M.cost = 1/M.score
    sol = M.localsolve("mosek")
    print sol.summary()
    