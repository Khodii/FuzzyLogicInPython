# Fuzzy Controller für Smarte Regelungen
# Aktives Reduzieren von Wind induzierten Vibrationen bei Hochhäusern

import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import matplotlib.pyplot as plt

"""Parameters of the first building"""
FIRST_M1 = 1.53 * 10**8
FIRST_K1 = 1.53 * 10**8
FIRST_B1 = 3.06 * 10**6
FIRST_M2 = 3.06 * 10**6
FIRST_K2 = 2.55 * 10**6
FIRST_B2 = 2.01 * 10**5

"""Parameters of the second building"""
SECOND_M1 = 1.81 * 10**7
SECOND_K1 = 1.81 * 10**7
SECOND_B1 = 3.62 * 10**5
SECOND_M2 = 3.62 * 10**5
SECOND_K2 = 3.01 * 10**5
SECOND_B2 = 0.238 * 10**5

"""Deklarieren von globalen Variablen"""
controlforce = 0
sim = 0
system = 0


def dx(m1, m2, k1, k2, b1, b2, u, fw, x, t):
    """
    Derivative of x(the state)
    ...
    Parameters
    ----------
    m1 : float
        mass of the building
    m2 : float
        mass of the TMD
    k1 : float
        lateral stiffness of the building
    k2 : float
        spring constant of the TMD
    b1 : float
        damping constant of the building
    b2 : float
        damping constant of the TMD
    u : function that takes a numpy vector of length 4 as input and returns a float
        state-dependent control force function u(x)
    fw : function that takes a float as input and returns a float
        time-dependent wind force function fw(t)
    x : numpy vector of length 4
        current state x = (y1, z, dy1, dz) where y1 = displacement of the building, y2 = displacement of TMD, z = y2 - y1

    t : float
        time

    Returns
    -------
    numpy vector of length 4
        derivative of x
    """
    A = np.array([[0, 0, 1, 0],
                 [0, 0, 0, 1],
                 [-k1/m1, k2/m1, -b1/m1, b2/m1],
                 [k1/m1, -k2*(1/m1+1/m2), b1/m1, -b2*(1/m1+1/m2)]])

    B1 = np.array([0, 0, -1/m1, 1/(m1 + m2)])
    B2 = np.array([0, 0, 1/m1, -1/m1])

    return A.dot(x) + u(x) * B1 + fw(t) * B2


def heun_step(m1, m2, k1, k2, b1, b2, u, fw, x, t, delta_t):
    """
    Calculate one integration step with Heun's method (explicit 2nd order)
    ...
    Parameters
    ----------
    m1 : float
        mass of the building
    m2 : float
        mass of the TMD
    k1 : float
        lateral stiffness of the building
    k2 : float
        spring constant of the TMD
    b1 : float
        damping constant of the building
    b2 : float
        damping constant of the TMD
    u : function that takes a numpy vector of length 4 as input and returns a float
        state-dependent control force function u(x)
    fw : function that takes a float as input and returns a float
        time-dependent wind force function fw(t)
    x : numpy vector of length 4
        current state x = (y1, z, dy1, dz) where y1 = displacement of the building, y2 = displacement of TMD, z = y2 - y1
    t : float
        time
    delta_t : float
        time step

    Returns
    -------
    new_x : numpy vector of length 4
        next state vector
    new_t : float
        next time
    """

    first_step = delta_t * dx(m1, m2, k1, k2, b1, b2, u, fw, x, t)
    next_step = delta_t * dx(m1, m2, k1, k2, b1, b2, u, fw, x + first_step, t + delta_t)
    new_x = x + 1/2 * (first_step + next_step)
    new_t = t + delta_t

    return new_x, new_t

def simulation(m1, m2, k1, k2, b1, b2, u, fw, x0, t0, delta_t, t_end):
    """
    Simulate behaviour of the building
    ...
    Parameters
    ----------
    m1 : float
        mass of the building
    m2 : float
        mass of the TMD
    k1 : float
        lateral stiffness of the building
    k2 : float
        spring constant of the TMD
    b1 : float
        damping constant of the building
    b2 : float
        damping constant of the TMD
    u : function that takes a numpy vector of length 4 as input and returns a float
        state-dependent control force function u(x)
    fw : function that takes a float as input and returns a float
        time-dependent wind force function fw(t)
    x0 : numpy vector of length 4
        initial state x0 = (y1, z, dy1, dz) where y1 = displacement of the building, y2 = displacement of TMD, z = y2 - y1
    t0 : float
        initial time
    delta_t : float
        time step
    t_end : float
        the simulation stops after reaching this time

    Returns
    -------
    x_values : numpy matrix (4 x number of time steps)
        the columns of this matrix are the states at every simulation step
    t_values : numpy vector of length equal to the number of time steps
        time values for every simulation step
    """

    x_list = []
    t_list = []

    x = x0
    t = t0
    while t < t_end:
        x, t = heun_step(m1, m2, k1, k2, b1, b2, u, fw, x, t, delta_t)
        x_list.append(x)
        t_list.append(t)

    return np.array(x_list).T, np.array(t_list)

def wind_force(t):
    """
    time-dependent wind force
    ...
    Parameters
    ----------
    t : float
        time

    Returns
    -------
    float
        wind force
    """
    p = 44444.0
    w = 1.0

    wt = w * t
    return p * (3*np.sin(wt) + 7*np.sin(2*wt) + 5*np.sin(3*wt) + 4*np.sin(4*wt))

def zero_control_force(x):
    """
    this function returns 0 and is used to simulate the uncontrolled behaviour of the building
    """
    return 0

def membership_functions():
    """
    Definieren der Membership Functions sowie deren Wertebereiche.
    Zusätzlich werden die Regeln der 7x7 Matrix implementiert.
    """

    global sim
    global system
    global controlforce

    # Eigene Membership Functions benennen
    names = ['NL', 'NM', 'NS', 'ZR', 'PS', 'PM', 'PL']

    # Create the three fuzzy variables - two inputs, one output
    #Input
    displacement = ctrl.Antecedent(np.linspace(-0.15, 0.15, 200), 'Displacement')
    velocity = ctrl.Antecedent(np.linspace(-1, 1, 200), 'Velocity')
    #displacement = ctrl.Antecedent(np.array([-5, -4.5, -3, -2, -1, 0, 1, 2, 3, 3.5, 4, 5, 5.5]), 'Displacement')
    #velocity = ctrl.Antecedent(np.array([-0.4, -0.3, -0.2, -0.1, -0.01, 0, 0.01, 0.1, 0.2, 0.3, 0.4]), 'Velocity')
    #Output
    controlforce = ctrl.Consequent(np.linspace(-8, 8, 200), 'Control Force')

    # Auto-membership function population is possible with .automf(3, 5, or 7)
    # Hier mit der vorgegebenen Liste names befüllt, siehe oben
    #displacement.automf(names=names)
    #velocity.automf(names=names)
    #controlforce.automf(names=names)

    #Individuelle Definition der Membership Functions

    #Control Force
    controlforce['NL'] = fuzz.zmf(controlforce.universe, -5, -4.5)
    controlforce['NM'] = fuzz.trimf(controlforce.universe, [-5.1, -4, -2.9])
    controlforce['NS'] = fuzz.trimf(controlforce.universe, [-3.1, -2, -0.9])
    controlforce['ZR'] = fuzz.trimf(controlforce.universe, [-0.95, 0, 0.95])
    controlforce['PS'] = fuzz.trimf(controlforce.universe, [0.9, 2, 3.1])
    controlforce['PM'] = fuzz.trimf(controlforce.universe, [2.9, 4, 5.1])
    controlforce['PL'] = fuzz.smf(controlforce.universe, 4.5, 5)

    #Velocity
    velocity['NL'] = fuzz.zmf(velocity.universe, -0.4, -0.3)
    velocity['NM'] = fuzz.trimf(velocity.universe, [-0.4, -0.2, -0.1])
    velocity['NS'] = fuzz.trimf(velocity.universe, [-0.2, -0.1, 0])
    velocity['ZR'] = fuzz.trimf(velocity.universe, [-0.02, 0, 0.02])
    velocity['PS'] = fuzz.trimf(velocity.universe, [0, 0.1, 0.2])
    velocity['PM'] = fuzz.trimf(velocity.universe, [0.1, 0.2, 0.4])
    velocity['PL'] = fuzz.smf(velocity.universe, 0.3,  0.4)


    #Displacement
    displacement['NL'] = fuzz.zmf(displacement.universe, -0.06, -0.04)
    displacement['NM'] = fuzz.trimf(displacement.universe, [-0.06, -0.035, -0.01])
    displacement['NS'] = fuzz.trimf(displacement.universe, [-0.025, -0.0125, 0])
    displacement['ZR'] = fuzz.trimf(displacement.universe, [-0.005, 0, 0.005])
    displacement['PS'] = fuzz.trimf(displacement.universe, [0, 0.0125, 0.025])
    displacement['PM'] = fuzz.trimf(displacement.universe, [0.01, 0.035, 0.06])
    displacement['PL'] = fuzz.smf(displacement.universe, 0.04, 0.06)

    displacement.view()
    velocity.view()
    controlforce.view()

    rule1 = ctrl.Rule(displacement['NL'] & velocity['NL'], consequent=controlforce['PL'])
    #rule1.view()
    rule2 = ctrl.Rule(displacement['NL'] & velocity['NM'], consequent=controlforce['PL'])
    rule3 = ctrl.Rule(displacement['NL'] & velocity['NS'], consequent=controlforce['PL'])
    rule4 = ctrl.Rule(displacement['NL'] & velocity['ZR'], consequent=controlforce['PL'])
    rule5 = ctrl.Rule(displacement['NL'] & velocity['PS'], consequent=controlforce['PL'])
    rule6 = ctrl.Rule(displacement['NL'] & velocity['PM'], consequent=controlforce['PL'])
    rule7 = ctrl.Rule(displacement['NL'] & velocity['PL'], consequent=controlforce['PL'])

    rule8 = ctrl.Rule(displacement['NM'] & velocity['NL'],  consequent=controlforce['ZR'])
    rule9 = ctrl.Rule(displacement['NM'] & velocity['NM'],  consequent=controlforce['PS'])
    rule10 = ctrl.Rule(displacement['NM'] & velocity['NS'], consequent=controlforce['PS'])
    rule11 = ctrl.Rule(displacement['NM'] & velocity['ZR'], consequent=controlforce['PS'])
    rule12 = ctrl.Rule(displacement['NM'] & velocity['PS'], consequent=controlforce['PS'])
    rule13 = ctrl.Rule(displacement['NM'] & velocity['PM'], consequent=controlforce['PM'])
    rule14 = ctrl.Rule(displacement['NM'] & velocity['PL'], consequent=controlforce['PM'])

    rule15 = ctrl.Rule(displacement['NS'] & velocity['NL'], consequent=controlforce['ZR'])
    rule16 = ctrl.Rule(displacement['NS'] & velocity['NM'], consequent=controlforce['ZR'])
    rule17 = ctrl.Rule(displacement['NS'] & velocity['NS'], consequent=controlforce['PS'])
    rule18 = ctrl.Rule(displacement['NS'] & velocity['ZR'], consequent=controlforce['ZR'])
    rule19 = ctrl.Rule(displacement['NS'] & velocity['PS'], consequent=controlforce['PS'])
    rule20 = ctrl.Rule(displacement['NS'] & velocity['PM'], consequent=controlforce['PS'])
    rule21 = ctrl.Rule(displacement['NS'] & velocity['PL'], consequent=controlforce['PS'])

    rule22 = ctrl.Rule(displacement['ZR'] & velocity['NL'], consequent=controlforce['NS'])
    rule23 = ctrl.Rule(displacement['ZR'] & velocity['NM'], consequent=controlforce['ZR'])
    rule24 = ctrl.Rule(displacement['ZR'] & velocity['NS'], consequent=controlforce['ZR'])
    rule25 = ctrl.Rule(displacement['ZR'] & velocity['ZR'], consequent=controlforce['ZR'])
    rule26 = ctrl.Rule(displacement['ZR'] & velocity['PS'], consequent=controlforce['ZR'])
    rule27 = ctrl.Rule(displacement['ZR'] & velocity['PM'], consequent=controlforce['ZR'])
    rule28 = ctrl.Rule(displacement['ZR'] & velocity['PL'], consequent=controlforce['PS'])

    rule29 = ctrl.Rule(displacement['PS'] & velocity['NL'], consequent=controlforce['NS'])
    rule30 = ctrl.Rule(displacement['PS'] & velocity['NM'], consequent=controlforce['NS'])
    rule31 = ctrl.Rule(displacement['PS'] & velocity['NS'], consequent=controlforce['NS'])
    rule32 = ctrl.Rule(displacement['PS'] & velocity['ZR'], consequent=controlforce['ZR'])
    rule33 = ctrl.Rule(displacement['PS'] & velocity['PS'], consequent=controlforce['NS'])
    rule34 = ctrl.Rule(displacement['PS'] & velocity['PM'], consequent=controlforce['ZR'])
    rule35 = ctrl.Rule(displacement['PS'] & velocity['PL'], consequent=controlforce['ZR'])

    rule36 = ctrl.Rule(displacement['PM'] & velocity['NL'], consequent=controlforce['NM'])
    rule37 = ctrl.Rule(displacement['PM'] & velocity['NM'], consequent=controlforce['NM'])
    rule38 = ctrl.Rule(displacement['PM'] & velocity['NS'], consequent=controlforce['NS'])
    rule39 = ctrl.Rule(displacement['PM'] & velocity['ZR'], consequent=controlforce['NS'])
    rule40 = ctrl.Rule(displacement['PM'] & velocity['PS'], consequent=controlforce['NS'])
    rule41 = ctrl.Rule(displacement['PM'] & velocity['PM'], consequent=controlforce['NS'])
    rule42 = ctrl.Rule(displacement['PM'] & velocity['PL'], consequent=controlforce['ZR'])

    rule43 = ctrl.Rule(displacement['PL'] & velocity['NL'], consequent=controlforce['NL'])
    rule44 = ctrl.Rule(displacement['PL'] & velocity['NM'], consequent=controlforce['NL'])
    rule45 = ctrl.Rule(displacement['PL'] & velocity['NS'], consequent=controlforce['NL'])
    rule46 = ctrl.Rule(displacement['PL'] & velocity['ZR'], consequent=controlforce['NL'])
    rule47 = ctrl.Rule(displacement['PL'] & velocity['PS'], consequent=controlforce['NL'])
    rule48 = ctrl.Rule(displacement['PL'] & velocity['PM'], consequent=controlforce['NL'])
    rule49 = ctrl.Rule(displacement['PL'] & velocity['PL'], consequent=controlforce['NL'])

    system = ctrl.ControlSystem(rules=[rule1, rule2, rule3, rule4, rule5, rule6, rule7,rule8, rule9,
                                        rule10, rule11, rule12, rule13, rule14, rule15,rule16, rule17,
                                        rule18, rule19, rule20, rule21, rule22,rule23, rule24, rule25,
                                        rule26, rule27, rule28, rule29, rule30, rule31, rule32, rule33,
                                        rule34, rule35, rule36, rule37,rule38, rule39, rule40, rule41,
                                        rule42, rule43, rule44, rule45,rule46, rule47, rule48, rule49])

    sim = ctrl.ControlSystemSimulation(system)


def fuzzy_control_force(x):
    """
    Benutzt die vorhergehenden Membership Functions und Regelbasis um das System zu regeln.
    Dafür wird x und x_punkt als Eingabe für Displacement und Velocity gegeben und die zugehörige
    Control Force mit Hilfe der Regelbasis returned.
    """
    global sim
    sim = ctrl.ControlSystemSimulation(system)
    sim.input['Displacement'] = x[0]  # x[0] = y1
    sim.input['Velocity'] = x[2]   # x[2] = dy1

    sim.compute()
    # Einheit steht nicht im Paper drin, 10^6 geraten
    # Negatives gibt bessere Ergebnisse, also wird es wahrscheinlich richtig sein xD
    return -sim.output['Control Force'] * 10**6


if __name__ == '__main__':

    membership_functions()

    x1, t1 = simulation(FIRST_M1, FIRST_M2, FIRST_K1, FIRST_K2, FIRST_B1, FIRST_B2, zero_control_force, wind_force,
                        np.zeros(4), 0, 0.1, 350.0)
    x2, t2 = simulation(SECOND_M1, SECOND_M2, SECOND_K1, SECOND_K2, SECOND_B1, SECOND_B2, zero_control_force,
                        wind_force, np.zeros(4), 0, 0.1, 350.0)

    x3, t3 = simulation(FIRST_M1, FIRST_M2, FIRST_K1, FIRST_K2, FIRST_B1, FIRST_B2, fuzzy_control_force, wind_force,
                      np.zeros(4), 0, 0.1, 350.0)

    x4, t4 = simulation(SECOND_M1, SECOND_M2, SECOND_K1, SECOND_K2, SECOND_B1, SECOND_B2, fuzzy_control_force,
                        wind_force, np.zeros(4), 0, 0.1, 350.0)

    plt.figure()
    plt.title('First building uncontrolled')
    plt.ylabel('Displacement [mm]')
    plt.xlabel('Time [s]')
    plt.plot(t1, 1000 * x1[0, :])

    plt.figure()
    plt.title('Second building uncontrolled')
    plt.ylabel('Displacement [mm]')
    plt.xlabel('Time [s]')
    plt.plot(t2, 1000 * x2[0, :])

    plt.figure()
    plt.title('Wind force')
    plt.ylabel('Wind force [N]')
    plt.xlabel('Time [s]')
    plt.plot(t1, wind_force(t1))

    plt.figure()
    plt.title('Displacement after Fuzzy Logic(1)')
    plt.ylabel('Displacement[mm]')
    plt.xlabel(' Time [s]')
    plt.plot(t3, 1000 * x3[0, :])

    plt.figure()
    plt.title('Displacement after Fuzzy Logic(2)')
    plt.ylabel('Displacement[mm]')
    plt.xlabel(' Time [s]')
    plt.plot(t4, 1000 * x4[0, :])

    plt.show()
