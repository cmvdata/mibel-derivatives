# CONTEXT — mibel-derivatives

## Objetivo del módulo

Construir un marco de valoración y análisis de sensibilidades para derivados energéticos sobre el mercado eléctrico ibérico (MIBEL), aplicable a productos representativos del libro de un Middle Office de utility o trading house: swing options, tolling agreements sobre CCGT y PPAs sobre generación solar.

El módulo entrega valoraciones coherentes con precios de mercado observables (OMIP forward, OMIE spot, MIBGAS) y sensibilidades de primer orden (delta, vega, gamma, theta) calculadas por shocks finitos. El análisis de riesgo agregado (VaR, ES, PFE, CVA) corresponde al módulo `mibel-risk`, que toma este módulo como input.

## Alcance temporal

- Calibración con datos históricos enero 2019 a diciembre 2024.
- Valoraciones de productos con vencimientos hasta diciembre 2030.

## Arquitectura de modelos

### Pieza 1: modelo del spot

Modelo de reversión a la media con saltos (mean-reverting jump-diffusion) sobre el log-precio spot horario:

dlogS_t = κ(θ_t − logS_t)dt + σdW_t + J·dN_t

donde θ_t incorpora estacionalidad determinista (componentes anual, semanal, diaria), J es el tamaño de salto (normal o doble exponencial), N_t es proceso de Poisson de intensidad λ.

Calibración por máxima verosimilitud sobre residuos detectados como no-salto, con detección previa de saltos por bipower variation o threshold.

### Pieza 2: modelo de la curva forward

Modelo Schwartz-Smith de dos factores estocásticos más componente determinista estacional:

logF(t, T) = X_t · e^(−κ(T−t)) + L_t + s(T)

donde X_t es el factor de corto plazo con reversión a la media, L_t es el factor de largo plazo (random walk), s(T) es la función estacional determinista.

Calibración por filtro de Kalman + máxima verosimilitud sobre datos OMIP forward, condicionando a reproducir tanto el spot OMIE actual como la curva forward OMIP actual.

### Versión 2 (future work)

Tercer factor estocástico estacional, prima ibérica como proceso, stochastic volatility.

## Productos a valorar

### Swing option

Contrato anual sobre electricidad MIBEL con strike fijo, límite diario de ejercicio y volumen máximo anual. Valoración por Longstaff-Schwartz Monte Carlo.

### Tolling agreement sobre Castejón I

Central de ciclo combinado de Iberdrola en Navarra. Datos públicos de referencia según Declaración Ambiental Iberdrola 2024 y benchmarks técnicos públicos (NREL, Aurecon/AEMO):

| Parámetro | Valor | Fuente |
|---|---|---|
| Potencia bruta diseño | 386,10 MW | Declaración Ambiental Iberdrola 2024 |
| Potencia neta estimada | ~370 MW | Bruta menos auxiliares (~4%) |
| Mínimo técnico estimado | ~120 MW (~30% Pmax) | Benchmark Aurecon configuración 1+1 |
| Heat rate a plena carga | 6,55 GJ/MWh | Derivado de rendimiento 55% Iberdrola |
| Heat rate a mínimo | ~7,5 GJ/MWh | Extrapolación Aurecon |
| Coste arranque caliente | ~50 €/MW | Benchmark Aurecon convertido EUR |
| Coste arranque templado | ~80 €/MW | Benchmark Aurecon convertido EUR |
| Coste arranque frío | ~110 €/MW | Benchmark Aurecon convertido EUR |
| Tiempo mínimo operación | 4-6 horas | Estándar CCGT moderna |
| Tiempo mínimo entre arranques | 2-4 horas | Estándar CCGT moderna |
| Ramp rate | 8 MW/min | Estándar CCGT monoeje |

Todos los parámetros se versionan en la tabla `asset_parameters` con campo `source` indicando si son públicos, contractuales, declarados o estimados.

Modelo de gas:

- MIBGAS como mercado principal (curva forward corta y media).
- TTF como referencia para curva forward larga (más de 3 años, por liquidez).
- Modelo de spot gas: mean-reverting jump-diffusion separado, calibrado de forma análoga al de electricidad.
- Correlación electricidad-gas calibrada sobre residuos.

Optimizador de operación: programación dinámica con estados (desacoplado, arrancando, mínimo técnico, producción óptima, parando) y restricciones temporales (TMO, TMA). Valoración por Longstaff-Schwartz sobre el spark spread, respetando restricciones operativas.

### PPA solar

Planta solar de referencia: 100 MW en Andalucía, perfil horario simulado con componente estacional anual + variabilidad estocástica.

Estructura: 80% de generación esperada vendida a precio fijo, 20% al spot. Sin caps ni floors en versión 1.

Cálculo del capture price hora a hora: ratio entre revenue real (generación × precio spot horario) y revenue baseline (generación × precio medio).

Valoración del swap implícito y análisis de sensibilidad a cannibalización solar.

## Parámetros numéricos

- Trayectorias Monte Carlo: 50.000 para valoraciones reportadas, 10.000 para desarrollo y tests.
- Curva de descuento: EURIBOR + OIS hasta 5 años, extrapolación constante más allá.
- Semilla aleatoria fija en tests para reproducibilidad.

## Anclaje regulatorio

El módulo se construye con consciencia del marco regulatorio español. Las inflexibilidades operativas se modelan según la conceptualización estándar en mercados eléctricos (Osinergmin Perú 2019, RIA 001-2019/V4, Vilches et al.) y se adaptan al marco español:

- Mercado diario e intradiario: OMIE, algoritmo Euphemia.
- Servicios de ajuste y restricciones técnicas: Red Eléctrica, procedimientos de operación PO 3.1, PO 3.2, PO 7.2, PO 7.3, PO 9.1, PO 9.2, PO 14.4.
- Supervisión: CNMC + ACER bajo REMIT.
- Reforma CNMC marzo 2024 (BOE-A-2024-6215): parametrización explícita de costes de arranque frío/caliente en restricciones técnicas.

Caso de referencia sobre comportamiento estratégico documentado: sanción CNMC a Neuro Energía y Gestión, SNC/DE/017/23, octubre 2024, por manipulación del mercado intradiario continuo bajo artículo 5 de REMIT (Vilches 2026).

## Estructura de tablas maestras

Cinco tablas de control para auditabilidad:

| Tabla | Campos mínimos |
|---|---|
| `asset_parameters` | Pmax, Pmin, heat rate, rampas, start costs, min up/down, fuente, versión |
| `market_prices` | Precio energía, gas, CO2, servicios, timestamp, fuente |
| `dispatch_states` | Estado, potencia, arranque/parada, activación, restricción |
| `settlement_reconciliation` | Programa, medida, liquidación, ajustes, desvíos, diferencias |
| `compliance_log` | Causa de cambio, usuario, hora, evidencia, versión de modelo |

## Productos del módulo

- Notebooks de calibración con validación (spot, forward, gas).
- Notebooks de valoración con sensibilidades por producto.
- Reporte consolidado en `reports/consolidated_report.md` con valoraciones, sensibilidades, análisis de escenarios.
- Bundle auditable: `scripts/_dump_code_bundle.py` con discovery por glob (no lista fija) y `code_bundle.txt` en `.gitignore`.

## Bibliografía base

- Schwartz, E.S. (1997). "The Stochastic Behavior of Commodity Prices: Implications for Valuation and Hedging." Journal of Finance.
- Schwartz, E.S. y Smith, J.E. (2000). "Short-Term Variations and Long-Term Dynamics in Commodity Prices." Management Science.
- Lucia, J.J. y Schwartz, E.S. (2002). "Electricity Prices and Power Derivatives: Evidence from the Nordic Power Exchange." Review of Derivatives Research.
- Longstaff, F.A. y Schwartz, E.S. (2001). "Valuing American Options by Simulation: A Simple Least-Squares Approach." Review of Financial Studies.
- Federico, G., Vives, X., Fabra, N. (varios). Literatura sobre poder de mercado en MIBEL.
- Vilches, C. (2026). "Detecting Structural Manipulation in Electricity Markets: The Neuro Case." Universitat de Barcelona.
- Osinergmin Perú (2019). RIA 001-2019/V4 sobre supervisión de inflexibilidades operativas en el SEIN.
- Red Eléctrica de España. Procedimientos de Operación del sistema eléctrico.
- BOE-A-2024-6215. Resolución CNMC 6 marzo 2024 sobre modificación de procedimientos de operación.
- BOE-A-2022-15755. Resolución CNMC sobre servicios de no frecuencia.
- BOE-A-2023-8113. Resolución CNMC 16 marzo 2023 sobre PO 3.8 y PO 9.2.
- NREL. "Power Plant Cycling Costs."
- AEMO/ElectraNet/Aurecon. "Generator Technical and Cost Parameters."
- Iberdrola España. Declaración Ambiental Central de Ciclo Combinado de Castejón 2024.

## Lo que NO hace este módulo

- No hace trading propiamente dicho (módulo `mibel-trading`).
- No hace VaR/ES/PFE/CVA (módulo `mibel-risk`).
- No optimiza la composición del portfolio.
- No hace hedging dinámico.
