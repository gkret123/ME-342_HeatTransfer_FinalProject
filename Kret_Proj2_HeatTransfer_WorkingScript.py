"""
Authors: Gabriel Kret, Maria Alvarado, Jonas Margono

ME-342: Heat Transfer
Course Instructor: Dr. Kamau Wright
Course Textbook: Fundamentals of Heat and Mass Transfer, 8th Edition, Bergman et. al.
Project 2: Heat Transfer Rod [Fastest Cooling and Highest q]

This code caluculates the heat transfer rates for various shapes in varying convection conditions(external flow, internal flow, free convection).
All table refereneces are from the course textbook.
"""

import math
import numpy as np
from scipy.optimize import minimize

# Given Temperatures (K) from problem statement
T_s = 100 + 273.15 # surface temperature
T_inf = 20 + 273.15 # Ambient temperature

# Constants
g = 9.81 #gravity m/s^2
Pr = 0.702 #Prandtl number for air from table A.4

# Geometry
L = 0.1524 # Length, m
D = 0.009525 # Diameter, m

# Thermal diffusivity of air at 300 K and 350 K (needed for linear interpolation between 300 K and 350 K values)
alpha_300 = 22.5e-6
alpha_350 = 29.9e-6
# Dynamic viscosity of air (needed for linear interpolation between 300 K and 350 K values)
visc_300 = 15.89e-6 # Pa-s at 300 K
visc_350 = 20.92e-6 # Pa-s at 350 K
# Thermal conductivity of air (needed for linear interpolation between 300 K and 350 K values)
k_air_300 = 26.3e-3 # W/m-K at 300 K
k_air_350 = 30e-3 # W/m-K at 350 K

#All shapes will have the same surface area
# Surface area of the cylinder (exclude the ends)
A = math.pi * D * L

#Properties of air at the film temperature
def film_properties(T_s, T_inf):
    T_f = (T_s + T_inf) / 2
    alpha = alpha_300 + (T_f - 300) * (alpha_350 - alpha_300) / (350 - 300)
    k_air = k_air_300 + (T_f - 300) * (k_air_350 - k_air_300) / (350 - 300)
    visc = visc_300 + (T_f - 300) * (visc_350 - visc_300) / (350 - 300)
    beta = 1.0 / T_f
    return T_f, alpha, k_air, visc, beta

#PART 1: Calculate the heat transfer rate for a cylinder in free convection

def cylinder_convection(T_s, T_inf, D, L, alpha, k_air, visc, beta):
    raD = g * beta * (T_s -T_inf) * D**3 / (visc * alpha) # Rayleigh number based on diameter
    vol = np.pi * (D/2)**2 * L # Volume of the cylinder
    # Empirical constants (c,n) for cylinder from table 9.1, these are based on the Rayleigh number
    if raD <= 10**-2:
        c, n = 0.675, 0.058
    elif raD > 10**-2 and raD <= 10**2:
        c, n = 1.02, 0.148
    elif raD > 10**2 and raD <= 10**4:
        c, n = 0.85, 0.188
    elif raD > 10**4 and raD <= 10**7:
        c, n = 0.480, 0.250
    else: # raD > 10^7
        c, n = 0.125, 0.333

    nu = c * raD**n # Nusselt number for cylinder in free convection
    h = nu * k_air / D # convective heat transfer coefficient w/m^2-K
    q = h * A * (T_s - T_inf) # heat transfer rate, W
    return A, raD, nu, h, q, vol


#PART 2: Calculate the heat transfer rate for a square cylinder in free convection

#The square cylinder is treated as 4 plates, 2 vertical, 1 top plate, and 1 bottom plate (see Table 9.3).
def square_convection(T_s, T_inf, A, alpha, k_air, visc, beta):
    w = A / 4 / L # width of the square cylinder, this is the characteristic length
    vol = w**2 * L # Volume of the square cylinder
    raW = g * beta * (T_s - T_inf) * w**3 / (visc * alpha) # Rayleigh number based on width
    Gr = g * beta * (T_s - T_inf) * w**3 / (visc**2) # Grashof number based on width
# 2 vertical plates
    nu_vert = (0.825 + (0.387 * raW**(1/6) / ((1 + (0.492 / Pr)**(9/16)))**(8/27)))**2 # Nusselt number for vertical square cylinder
    h_vert = nu_vert * k_air / w # convective heat transfer coefficient w/m^2-K
    q_vertical = h_vert * A/4 * (T_s - T_inf) # heat transfer rate for vertical plates, W (Use A/4 bc each plate faces 1/4 of the total area)
# 1 top plate
    nu_top = 0.54 * raW**(1/4) # Nusselt number correlation for top plate
    h_top = nu_top * k_air / w # convective heat transfer coefficient w/m^2-K
    q_top = h_top * A/4 * (T_s - T_inf) # heat transfer rate for top plate, W (Use A/4 bc each plate faces 1/4 of the total area)
# 1 bottom plate
    nu_bottom = 0.52 * raW**(1/5) # Nusselt number correlation for bottom plate
    h_bottom = nu_bottom * k_air / w # convective heat transfer coefficient w/m^2-K
    q_bottom = h_bottom * A/4 * (T_s - T_inf) # heat transfer rate for bottom plate, W (Use A/4 bc each plate faces 1/4 of the total area)
    q_total = q_top + q_bottom + 2 * q_vertical # total heat transfer rate for square cylinder, W
    print(f"Nusselt number for vertical plates: {nu_vert:.3f}")
    print(f"Nusselt number for top plate: {nu_top:.3f}")
    print(f"Nusselt number for bottom plate: {nu_bottom:.3f}")
    return w, raW, Gr, h_vert, h_top, h_bottom, q_top, q_bottom, q_vertical, q_total, vol

#PART 3: Calculate the heat transfer rate for a Cone in free convection
#The cone is approximated as a circular base and N inclined plates at an angle theta (see Table 9.3).
# Cone volume optimizations
A_cone = A # Surface area of the cone (same as cylinder)

#We will test 2 conditions, one with the maximum cone volume and one with the minimum cone volume.
#In both cases we will use the same surface area as the cylinder.

#x = [r, l] # r = base radius, l = slant height
def neg_cone_volume(x, A_cone):
    r, l = x
    if l <= r:
        return np.inf  
    h = math.sqrt(l**2 - r**2)
    V = (1/3) * math.pi * r**2 * h
    return -V

#Equality constraint for the surface area of the cone
cons = [{
    'type': 'eq',
    'fun': lambda x: math.pi * x[0]**2 + math.pi * x[0] * x[1] - A_cone
}]

# Feasible initial guess (l solves the area constraint exactly)
r0 = 0.0005   # 5 mm
l0 = A_cone/(math.pi*r0) - r0
x0 = [r0, l0]

# Bounds on r,l
bnds = [(1e-8, 1), (1e-8, 1)]

#Call minimize, then inspect the result
res = minimize(
    fun=neg_cone_volume,
    x0=x0,
    args=(A_cone,),
    method='SLSQP',
    bounds=bnds,
    constraints=cons,
    options={'ftol':1e-12, 'maxiter':5000}
)

print(res)
if not res.success:
    raise RuntimeError(f"Optimization failed: {res.message}")

r_opt_max, l_opt_max = res.x
h_opt_max = math.sqrt(l_opt_max**2 - r_opt_max**2)
V_opt_max = (1/3)*math.pi*r_opt_max**2*h_opt_max
print("The shapes analyzed are a cylinder, square cylinder, and cone.")

print("while the correlations used for the cylinder and square are readily available, the cone requires some optimization.")
print("The first step is to find the optimal radius and slant height of the cone.")
print("We will use the same surface area as the cylinder and find the cone with the maximum internal volume (since the minimum is the degenerate case of 0 m^3).")

print("\nFOR THE CONE MAXIMIZATION CONDITION:")
print(f"  Optimal radius:        {r_opt_max:.6f} m")
print(f"  Optimal slant height: {l_opt_max:.6f} m")
print(f"  Optimal height:       {h_opt_max:.6f} m")
print(f"  Maximum volume:       {V_opt_max:.8e} m³")

def surface_area_constraint(A):
    return lambda x: math.pi * x[0]**2 + math.pi * x[0] * x[1] - A

#Free convection equations for the cone
def cone_natural_convection(T_s, T_inf, D, H, alpha, k_air, visc, beta, N_plates=360):
    R = D/2 # base radius of the cone
    s = np.sqrt(R**2 + H**2) #slant height of the cone
    A_lat = np.pi * R * s #lateral surface area of the cone, this is the area of the inclined plates
    A_base = np.pi * R**2 #base area of the cone
    theta = np.arcsin(R / s) #Calculate the angle of inclination of the plates
    #Inclined plate heat transfer rate
    Ra_angled = g * math.cos(theta) * beta * (T_s - T_inf) * s**3 / (visc * alpha) # Rayleigh number based on slant height
    Nu_angled = (0.825 + (0.387 * Ra_angled**(1/6) / ((1 + (0.492 / Pr)**(9/16)))**(8/27)))**2 #Nusselt number for inclined plates (Table 9.3)
    h_lat = Nu_angled * k_air / s # convective heat transfer coefficient w/m^2-K
    q_lat = N_plates * h_lat * A_lat/N_plates * (T_s - T_inf) # heat transfer rate for inclined plates, W
    # Base heat transfer rate
    Ra_base = g * beta * (T_s - T_inf) * D**3 / (visc * alpha) # Rayleigh number based on diameter
    Nu_base = 0.52 * Ra_base**(1/5) # Nusselt number for base (Table 9.3)
    h_base = Nu_base * k_air / D #heat transfer coefficient w/m^2-K
    q_base = h_base * A_base * (T_s - T_inf) # heat transfer rate for base, W
    return h_lat, q_lat, h_base, q_base, q_lat + q_base

def cone_natural_convection_cylinders(T_s, T_inf,
                                     D_base, H,
                                     alpha, k_air,
                                     visc, beta,
                                     N_segments=200):
    #Approximate a cone by N_segments vertical cylinders for free convection using each slice's own surface area.

    dz    = H / N_segments
    q_tot = 0.0
    A_tot = 0.0

    for i in range(N_segments):
        # 1) slice location & local diameter
        z    = (i + 0.5) * dz
        D_i  = D_base * (1 - z/H)
        if D_i <= 0:
            continue

        # 2) call your cylinder formula *only* to get h_i
        #    ignore its returned A and q.
        _, _, _, h_i, _, _ = cylinder_convection(
            T_s, T_inf,
            D_i, dz,
            alpha, k_air,
            visc, beta
        )

        #compute this slice's true area & q
        A_i   = math.pi * D_i * dz
        q_i   = h_i * A_i * (T_s - T_inf)

        q_tot += q_i
        A_tot += A_i

    # area‐weighted average h
    h_avg = q_tot / (A_tot * (T_s - T_inf))
    return h_avg, q_tot

# Main execution for CH9 (Free Convection) analysis
print("\n|----------CHAPTER 9 (FREE CONVECTION) ANALYSIS:----------|")
print(f"Film Temp: {film_properties(T_s, T_inf)[0]}")

T_f, alpha, k_air, visc, beta = film_properties(T_s, T_inf)
A, raD, nu, h, q, volume_cylinder = cylinder_convection(T_s, T_inf, D, L, alpha, k_air, visc, beta)
print(f"Surface area (A): {A:.3f} m²")
print("\n---RESULTS FOR A CYLINDRICAL CROSS SECTION (Ch9 - Free Convection):---")
print(f"Volume of the cylinder: {volume_cylinder:.8f} m³")
print(f"Rayleigh number (Ra_D): {raD:.3e}")
print(f"Nusselt number (Nu): {nu:.3f}")
print(f"Convection coefficient (h): {h:.3f} W/(m²·K)")
print(f"Total heat transfer rate (q_cylinder_total): {q} W")

print("\n---RESULTS FOR A SQUARE CROSS SECTION (Ch9 - Free Convection):---")
w, raW, Gr, h_vert, h_top, h_bottom, q_top, q_bottom, q_vertical, q_total, volume_square = square_convection(T_s, T_inf, A, alpha, k_air, visc, beta)
print(f"Width of the square cross section: {w:.8f} m")
print(f"Volume of the square cylinder: {volume_square:.8f} m³")
print(f"Rayleigh number (Ra_W): {raW:.3e}")
print(f"Grashof number (Gr): {Gr:.3e}")
print(f"Convection coefficient for vertical plates (h_vertical): {h_vert:.3f} W/(m²·K)")
print(f"Convection coefficient for upper plate (h_top): {h_top:.3f} W/(m²·K)")
print(f"Convection coefficient for lower plate (h_bottom): {h_bottom:.3f} W/(m²·K)")
print(f"Heat transfer rate for upper plate (q_top): {q_top:.3f} W")
print(f"Heat transfer rate for lower plate (q_bottom): {q_bottom:.3f} W")
print(f"Heat transfer rate for vertical plates (q_vertical): {q_vertical:.3f} W")
print(f"Total heat transfer rate (q_square_total): {q_total:.3f} W")




#calculate the needed parameters for cone, these values are needed for the free convection calculations
D_max = 2 * r_opt_max
L_max = math.sqrt(l_opt_max**2 - r_opt_max**2)
h_lat_max, q_lat_max, h_base_max, q_base_max, q_tot_max = cone_natural_convection(T_s, T_inf, D_max, l_opt_max, alpha, k_air, visc, beta)

print("\n---RESULTS FOR THE CONE WITH MAXIMIZED INTERNAL VOLUME (Ch9 - Free Convection):---")
print(f"Area of the cone: {A_cone:.3f} m²")
print(f"Lateral plates h = {h_lat_max:.3f} W/m²K,  q = {q_lat_max:.3f} W")
print(f"Base plate: h = {h_base_max:.3f} W/m²K,  q = {q_base_max:.3f} W")
print(f"Total Heat Transfer Rate for Cone with Maximized Volume (q_cone_max): {q_tot_max:.8f} W")

print("\nThe above results use a method of apporximating the cone as a series of inclined plates.")

print("\nThe following results use a method of approximating the cone as a series of N stacked cylinders.\n")
h_cyl_stack, q_cyl_stack = cone_natural_convection_cylinders(
    T_s, T_inf,
    D_max, l_opt_max,      # or D_min, H_min
    alpha, k_air,
    visc, beta,
    N_segments=360
)
print(f"Cone via stacked-cylinders: h = {h_cyl_stack:.3f}, q = {q_cyl_stack:.3f} W")
#End of Chapter 9 Analysis






#Chapter 7 Analysis
#This is the analysis for the EXTERNAL flow, we will use the same geometry as the above, but we will use the external flow equations.
#We will use the same surface area as the cylinder.

#T_inf = 20 degC, same as above so keep the same parapemeters
#T_s = 100 degC, same as above so keep the same parameters
V = 3 #avg velocity of the air, m/s

def get_fluid_props_Ch7(fluid, T_f):
    #Compute film temperature T_f and linearly interpolate fluid properties between two reference temperatures.
    #fluid can be 'air' or 'water'

    # e.g. filled from Table A.4 (air) and A.6 (water), "Property"1 is for T1, "Property"2 is for T2
    ref = {
        'air': {
            'T1': 300, 'T2': 350, #K
            'rho1': 1.1614,   'rho2': 0.9950,# kg/m^3
            'mu1': 184.6e-7, 'mu2': 208.2e-7,# N-s/m^2
            'k1' : 26.3e-3,   'k2' : 30.0e-3,# W/m-K
            'Pr1': 0.707,    'Pr2': 0.700
        },
        'water': {
            'T1': 330, 'T2': 335,
            'rho1': 1/(1.016e-3),   'rho2': 1/(1.018e-3),# kg/m3 rho = 1/spec.vol.
            'mu1': 489e-6, 'mu2': 453e-6, # N-s/m^2
            'k1' : 650e-3,    'k2' : 656e-3,# W/m-K
            'Pr1': 3.15,     'Pr2': 2.88 
        }
    }

    d = ref[fluid.lower()]
    def interp(prop):
        return d[prop+'1'] + (T_f - d['T1'])*(d[prop+'2']-d[prop+'1'])/(d['T2']-d['T1'])

    props = {
        'rho': interp('rho'),
        'mu' : interp('mu'),
        'k'  : interp('k'),
        'Pr' : interp('Pr')
    }
    return props

def forced_convection_cylinder(T_s, T_inf, D, L, V, fluid_props):
    #Churchill–Bernstein for cross‐flow over a cylinder:
      #Nu_D = 0.3 + (0.62 Re^0.5 Pr^(1/3) / [1 + (0.4/Pr)^(2/3)]^(1/4))* [1 + (Re / 282000)^(5/8)]^(4/5)

    rho, mu, k, Pr = fluid_props['rho'], fluid_props['mu'], fluid_props['k'], fluid_props['Pr']
    Re_D = rho * V * D / mu
    term1 = 0.62 * Re_D**0.5 * Pr**(1/3)
    term2 = (1 + (0.4/Pr)**(2/3))**0.25
    term3 = (1 + (Re_D/282000)**(5/8))**(4/5)
    Nu_D = 0.3 + term1/term2 * term3
    h = Nu_D * k / D
    A_surf = math.pi * D * L
    q = h * A_surf * (T_s - T_inf)
    return Re_D, Nu_D, h, q

def forced_convection_square(T_s, T_inf, A_total, L, V, fluid_props):
    #Cross‐flow over a square prism of side w = A_total/(4L), using the same Churchill–Bernstein form as the cylinder.

    rho, mu, k, Pr = (fluid_props[k] for k in ('rho','mu','k','Pr'))
    w = A_total / 4 / L
    Re_w = rho * V * w / mu

    # Churchill–Bernstein with D->w
    term1 = 0.62 * Re_w**0.5 * Pr**(1/3)
    term2 = (1 + (0.4/Pr)**(2/3))**0.25
    term3 = (1 + (Re_w/282000)**(5/8))**(4/5)
    Nu_w = 0.3 + term1/term2 * term3

    h = Nu_w * k / w
    A_surf = A_total   # all four faces
    q = h * A_surf * (T_s - T_inf)
    return Re_w, Nu_w, h, q


#Approximate a cone’s lateral surface as N small cylinders of decreasing diameter.
# The base diameter is D_base, the slant height is L_slant, and the velocity is V.
# The average heat transfer coefficient is calculated by summing the heat transfer rates of each cylinder and dividing by the total area.
def forced_convection_cone_stack(T_s, T_inf, D_base, L_slant, V, fluid_props, N=200):
    
    rho, mu, k, Pr = fluid_props['rho'], fluid_props['mu'], fluid_props['k'], fluid_props['Pr']
    print(f"Fluid properties: rho = {rho:.2f}, mu = {mu:.2e}, k = {k:.2f}, Pr = {Pr:.2f}")
    cone_height = np.sqrt(L_slant**2 - (D_base/2)**2)  # height of the cone
    dx = cone_height / N
    q_total = 0.0
    A_total = 0.0
    
    for i in range(N):
        #find diameter of the ith cylinder
        z = (i+0.5) * dx
        r_mid = D_base / 2 * (1 - z / cone_height)
        D_i = 2 * r_mid
        
        if D_i <= 0:
            continue
        
        # local cylinder area
        A_i = np.pi * D_i * dx
        
        # Re, Nu via Churchill–Bernstein
        Re_i = rho * V * D_i / mu
        term1 = 0.62 * Re_i**0.5 * Pr**(1/3)
        term2 = (1 + (0.4/Pr)**(2/3))**0.25
        term3 = (1 + (Re_i/282000)**(5/8))**(4/5)
        Nu_i = 0.3 + term1/term2 * term3
        
        h_i = Nu_i * k / D_i
        q_i = h_i * A_i * (T_s - T_inf)
        q_total += q_i
        A_total += A_i
    h_avg = q_total / (T_s - T_inf) / A_total  # average h from total q and area
    
    return h_avg, q_total



#Main execution for CH7 (Forced External Convection) analysis
print("\n|----------CHAPTER 7 (FORCED EXTERNAL CONVECTION) ANALYSIS:----------|")
fluid = "air"   # or "water"
props = get_fluid_props_Ch7(fluid, T_f)
print(f"Fluid properties in {fluid}: rho = {props['rho']:.2f} kg/m^3, mu = {props['mu']:.2e} N-s/m^2, k = {props['k']:.2f} W/m-K, Pr = {props['Pr']:.2f}")
# cylinder
Re_cyl, Nu_cyl, h_cyl, q_cyl = forced_convection_cylinder(T_s, T_inf, D, L, V, props)
print("---RESULTS FOR A CYLINDRICAL CROSS SECTION (Ch7 - Forced Convection):---")
print(f"Cylinder: Re = {Re_cyl:.4}, Nu = {Nu_cyl:.2f}, h = {h_cyl:.2f}, q = {q_cyl:.2f} W")

# square
Re_sq, Nu_sq, h_sq, q_sq = forced_convection_square(T_s, T_inf, A, L, V, props)
print("\n---RESULTS FOR A SQUARE CROSS SECTION (Ch7 - Forced Convection):---")
print(f"Square: Re = {Re_sq:.4}, Nu = {Nu_sq:.2f}, h = {h_sq:.2f}, q = {q_sq:.2f} W")

# cone (approx)
#D_max, L_max from optimization above in Ch9 analysis
h_cone_max, q_cone_max = forced_convection_cone_stack(T_s, T_inf, D_max, L_max, V, props)
print("\n---RESULTS FOR A CONE CROSS SECTION (Ch7 - Forced Convection):---")
print(f"Cone (max vol): h = {h_cone_max:.2f}, q = {q_cone_max:.2f} W")

#End of Chapter 7 Analysis





#Chapter 8 Analysis
#This is the analysis for the INTERNAL flow, we will use the same geometry as the above, but we will use the internal flow equations.
#We are not changing the geometry, the cylinder is assumed to be thin walled so we can use the same diameter as the above. i.e. D_inner = D_outer = D

#We will use the same surface area as the cylinder.
#We will use the same properties as the above, but we will use the internal flow equations.
def internal_forced_convection_cylinder(D_h, L, V, fluid_props):
    #Internal forced convection in a uniform duct of hydraulic diameter D_h and length L:
    #Laminar (Re<2300):  Nu = 3.66 eqn 8.55
    #Transitional (2300<Re<10000): Nu = (f/8)*(Re - 1000) * Pr / (1 + 12.7 * (f/8)**0.5 * (Pr**(2/3) - 1)) eqn 8.62
    #Turbulent (Re>=2300): Nu = 0.023 Re^0.8 Pr^0.4 --> eqn 8.60, this is a somewhat crude approximation for turbulent flow, since we are using it over 2300, but the formula applies to Re > 10,000
    
    rho, mu, k, Pr = fluid_props['rho'], fluid_props['mu'], fluid_props['k'], fluid_props['Pr']
    Re = rho * V * D_h / mu
    if Re < 2300:
        Nu = 3.66 # eqn 8.55 -- laminar flow, const. Nu for fully developed flow, const. T_surface = 100 degC
        #print(f"Laminar flow: Re = {Re:.2f}, Nu = {Nu:.2f}")
    elif Re > 2300 and Re < 10000: 
        f = (0.79 * np.log(Re) - 1.64)**(-2) # eqn 8.21 turb. flow friction factor. Re > 3000 and Re < 5e-6
        Nu = (f/8)*(Re - 1000) * Pr / (1 + 12.7 * (f/8)**0.5 * (Pr**(2/3) - 1)) # eqn 8.62, turbulent flow. Re > 3000 and Re < 5e-6, this is good in case we have an Re in the transitional range.
        #print(f"Transitional flow: Re = {Re:.2f}, Nu = {Nu:.2f}")   
    else:
        Nu = 0.023 * Re**0.8 * Pr**0.4  # eqn 8.60 -- turbulent flow, fully developed flow, T_surface > T_bulk so exponential term on Pr is 0.4
        #print(f"Turbulent flow: Re = {Re:.2f}, Nu = {Nu:.2f}")
    h = Nu * k / D_h 
    A_w = math.pi * D_h * L    # inner wetted perimeter × length
    q = h * A_w * (T_s - T_inf)
    return Re, Nu, h, q

def internal_forced_convection_square(A, L, V, fluid_props):
    #Internal forced convection in a uniform square duct of width w = A/(4*L) and length L:
    #Laminar (Re<2300):  Nu = 3.66 eqn 8.55
    #Transitional (2300<Re<10000): Nu = (f/8)*(Re - 1000) * Pr / (1 + 12.7 * (f/8)**0.5 * (Pr**(2/3) - 1)) eqn 8.62
    #Turbulent (Re>=2300): Nu = 0.023 Re^0.8 Pr^0.4 --> eqn 8.60, this is a somewhat crude approximation for turbulent flow, since we are using it over 2300, but the formula applies to Re > 10,000
    w = A / 4 / L # width of the square cylinder, this is the characteristic length
    rho, mu, k, Pr = fluid_props['rho'], fluid_props['mu'], fluid_props['k'], fluid_props['Pr']
    Re = rho * V * w / mu
    if Re < 2300:
        Nu = 2.98 # table 8.1. Nu = 2.98 becuase we are using the square (b=a) duct, this is a constant value for laminar fully developed flow, const. T_surface = 100 degC
        #print(f"Laminar flow: Re = {Re:.2f}, Nu = {Nu:.2f}")
    elif Re > 2300 and Re < 10000: #no table to reference for transitional flow in a square duct, so we will use the same equation as above for the cylinder with a hydralic diameter of w
        f = (0.79 * np.log(Re) - 1.64)**(-2) # eqn 8.21 turb. flow friction factor. Re > 3000 and Re < 5e-6
        Nu = (f/8)*(Re - 1000) * Pr / (1 + 12.7 * (f/8)**0.5 * (Pr**(2/3) - 1)) # eqn 8.62, turbulent flow. Re > 3000 and Re < 5e-6, this is good in case we have an Re in the transitional range.
        #print(f"Transitional flow: Re = {Re:.2f}, Nu = {Nu:.2f}")   
    else: # if Re > 10000, we can use the turbulent flow equation, no table to reference for turbulent flow in a square duct, so we will use the same equation as above for the cylinder with a hydralic diameter of w
        Nu = 0.023 * Re**0.8 * Pr**0.4  # eqn 8.60 -- turbulent flow, fully developed flow, T_surface > T_bulk so exponential term on Pr is 0.4
        #print(f"Turbulent flow: Re = {Re:.2f}, Nu = {Nu:.2f}")
    h = Nu * k / w 
    A_w = L * 4 * w    # inner wetted perimeter × length
    q = h * A_w * (T_s - T_inf)
    return Re, Nu, h, q

def internal_forced_convection_cone(D_base, L_slant, V, fluid_props, N=100):
    
    #Internal flow in a conical tube from diameter D_base down to zero:
    #Slice into N segments of length dx along the cone height.
    #Local D_i = D_base*(1 - z/H), H = sqrt(L_slant^2 - (D_base/2)^2).
    #Apply same laminar/transitional/turbulent rule/eqns as the circular cross section (see above) on each slice; sum q_i and A_i.
    #Use same method of breaking the cone into N segments as in the forced convection case.
   
    rho, mu, k, Pr = fluid_props['rho'], fluid_props['mu'], fluid_props['k'], fluid_props['Pr']
    H = math.sqrt(L_slant**2 - (D_base/2)**2) # height of the cone
    dx = H / N # height of each slice
    q_total = 0.0
    A_total = 0.0
    for i in range(N):
        z = (i + 0.5)*dx # mid-point of slice
        D_i = D_base * (1 - z/H) # diameter of the slice
        V_i = V * (D_i / D_base)**2 # velocity of the slice, this is a function of the diameter of the slice
        Re_i = rho * V_i * D_i / mu # Reynolds number for the slice
        if Re_i < 2300:
            Nu_i = 3.66 # eqn 8.55 -- laminar flow, const. Nu for fully developed flow, const. T_surface = 100 degC
            #print(f"Laminar flow: Re = {Re_i:.2f}, Nu = {Nu_i:.2f}")
        elif Re_i > 2300 and Re_i < 10000: 
            f = (0.79 * np.log(Re_i) - 1.64)**(-2) # eqn 8.21 turb. flow friction factor. Re > 3000 and Re < 5e-6
            Nu_i = (f/8)*(Re_i - 1000) * Pr / (1 + 12.7 * (f/8)**0.5 * (Pr**(2/3) - 1)) # eqn 8.62, turbulent flow. Re > 3000 and Re < 5e-6, this is good in case we have an Re in the transitional range.
            #print(f"Transitional flow: Re = {Re_i:.2f}, Nu = {Nu_i:.2f}")   
        else: # if Re > 10000, we can use the turbulent flow equation
            Nu_i = 0.023 * Re_i**0.8 * Pr**0.4  # eqn 8.60 -- turbulent flow, fully developed flow, T_surface > T_bulk so exponential term on Pr is 0.4
            #print(f"Turbulent flow: Re = {Re_i:.2f}, Nu = {Nu_i:.2f}")
        h_i = Nu_i * k / D_i
        A_i = math.pi * D_i * dx
        q_i = h_i * A_i * (T_s - T_inf)
        q_total += q_i
        A_total += A_i
        #print(f"Slice {i+1}: D_i = {D_i:.4f}, Re_i = {Re_i:.2f}, Nu_i = {Nu_i:.2f}, h_i = {h_i:.2f}, A_i = {A_i:.4f}, q_i = {q_i:.2f} W")
    h_avg = q_total / ((T_s - T_inf) * A_total)
    return h_avg, q_total

#Main execution for CH8 (Forced Internal Convection) analysis
print("\n|--------------CHAPTER 8 (FORCED INTERNAL CONVECTION) COMPARISON:------------|")
fluid = "air"   # or "water"
props = get_fluid_props_Ch7(fluid, T_f)
print(f"Fluid properties in {fluid}: rho = {props['rho']:.4f} kg/m^3, mu = {props['mu']:.4e} N-s/m^2, k = {props['k']:.4f} W/m-K, Pr = {props['Pr']:.4f}")
# Cylinder
Re_cyl_i, Nu_cyl_i, h_cyl_i, q_cyl_i = internal_forced_convection_cylinder(D, L, V, props)
print("---RESULTS FOR A CYLINDRICAL CROSS SECTION (Ch8 - Forced Convection):---")
print(f"Cylinder: Re = {Re_cyl_i:.5}, Nu = {Nu_cyl_i:.2f}, h = {h_cyl_i:.2f}, q = {q_cyl_i:.2f} W")

# Square
Re_sq_i, Nu_sq_i, h_sq_i, q_sq_i = internal_forced_convection_square(A, L, V, props)
print("\n---RESULTS FOR A SQUARE CROSS SECTION (Ch8 - Forced Convection):---")
print(f"Square:   Re = {Re_sq_i:.5}, Nu = {Nu_sq_i:.2f}, h = {h_sq_i:.2f}, q = {q_sq_i:.2f} W")

# Cone (use D_max & L_max from above Chapter 9 code)
h_cone_i, q_cone_i = internal_forced_convection_cone(D_max, L_max, V, props)
print("\n---RESULTS FOR A CONE CROSS SECTION (Ch8 - Forced Convection):---")
print(f"Cone: Re varies,  h_avg = {h_cone_i:.2f}, q = {q_cone_i:.2f} W")