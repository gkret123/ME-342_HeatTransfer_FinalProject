ME_342_HeatTransfer_FinalProj 

# Heat Transfer Analysis of Rod Geometries

## Authors
Gabriel Kret, Maria Alvarado, Jonas Margono  
**Course:** ME-342: Heat Transfer  
**Instructor:** Dr. Kamau Wright  
**Textbook:** *Fundamentals of Heat and Mass Transfer*, 8th Ed., Bergman et al.  
**Project Title:** Project 2: Heat Transfer Rod – *Fastest Cooling and Highest Heat Transfer Rate (q)*

---

## Project Objective

This project investigates which rod geometry—cylinder, square, or cone—achieves the highest convective heat transfer rate (`q`) under three conditions:
1. **Free (Natural) Convection**
2. **Forced External Convection**
3. **Forced Internal Convection**

All geometries are constrained to:
- Same **lateral surface area** (~0.005 m²)
- Same **length** (6 inches = 0.1524 m)
- Constant **surface temperature** of 100°C
- Exposed to air or water at 20°C

---

## 📌 Key Concepts

- Empirical Nusselt number correlations from *Fundamentals of Heat and Mass Transfer* were used to compute heat transfer coefficients.
- Cone shape required optimization to maximize volume while maintaining a constant surface area.
- For forced convection, the **Churchill–Bernstein correlation** was applied.
- For internal flow, **laminar, transitional, or turbulent** Nusselt number equations were used based on Reynolds number.
- Cone was approximated via discretized **cylinders (internal/forced convection)** or **inclined plates (free convection)**.

---

## 🧪 Methods & Tools

- **Language:** Python  
- **Libraries:** `NumPy`, `SciPy`, `math`  
- **Optimization:** `scipy.optimize.minimize` to maximize cone volume  
- **CFD (proposed for future work):** ANSYS Fluent for validating cone behavior under flow

---

## 📁 File Overview

- `Kret_Proj2_HeatTransfer_WorkingScript.py`: Complete analysis and comparison script
- `README.md`: Project summary and documentation (this file)
- `report.pdf`: Detailed report with theory, results, and analysis 

---

## 📊 Summary of Results

| Geometry | Free Convection (q) | Forced External Convection (q) | Forced Internal Convection (q) |
|----------|---------------------|--------------------------------|--------------------------------|
| Cylinder | 4.42 W              | 21.58 W                        | 4.03 W                         |
| Square   | **4.77 W**          | **24.30 W**                    | **4.18 W**                     |
| Cone     | 2.66 W              | 9.66 W                         | 2.27 W                         |

**Conclusion:** The **square rod** has the highest heat transfer rate in all cases, making it the most efficient geometry for thermal dissipation under the given constraints.

---

## References

- Bergman, T., Lavine, A., Incropera, F., & Dewitt, D. (2011). *Fundamentals of Heat and Mass Transfer* (8th ed.). Wiley.
- Janjua, M. M., et al. (2020). "Numerical Study of Forced Convection Heat Transfer..." *Journal of Thermal Analysis and Calorimetry*.
- Sparrow, E. M., et al. (2004). "Archival Correlations..." *International Journal of Heat and Mass Transfer*.
- Wang, X., et al. (2007). "Experimental Correlation of Forced Convection..." *Experimental Thermal and Fluid Science*.

---

