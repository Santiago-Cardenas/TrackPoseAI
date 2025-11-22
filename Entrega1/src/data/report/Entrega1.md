# Análisis de Actividades Humanas en Video

## Resumen
Se trabajó con **981 frames** de varios clips en formato MP4. La base inicial tuvo **136 columnas** (landmarks x/y/z + confidence/visibility y metadatos); tras enriquecer con nombres descriptivos y rasgos dinámicos (velocidades, magnitud y aceleración) se alcanzaron **204 columnas**.  
Distribución por clase: **caminar-atrás (242)**, **girar (223)**, **caminar-adelante (200)**, **sentarse (162)**, **levantarse (154)**. Velocidades: **lento (631)**, **rápido (350)**. **0 valores faltantes**.  
Los **11 landmarks críticos** analizados fueron nariz, hombros, muñecas, **caderas (23,24)**, **rodillas (25,26)** y **tobillos (27,28)**. Se observó **confianza muy alta en caderas (~0.998)** y menor en **rodillas/tobillos (~0.80–0.87)**. La **correlación en Y** fue muy alta entre pares de **piernas** (rodillas–tobillos ≈0.90–0.92) y menor entre **caderas** y piernas (≈0.27–0.49), sugiriendo que el **core** (caderas/torso) aporta señal informativa complementaria. Aunque puede que este resultado se haya visto afectado debido al uso de un pantalon suelto, que obstruye un poco la vista de rodillas y tobillos, al momento de grabar los videos.

---

## Preguntas de interés
1) ¿Qué **familias de rasgos** discriminan mejor las cinco actividades?  
2) ¿Qué **conjunto mínimo de landmarks** mantiene buen desempeño: (a) **todo el cuerpo**, (b) **caderas+tronco+hombros**, o (c) **caderas+rodillas+tobillos**?  
3) ¿Cómo influye la **categoría de velocidad** (lento/rápido) en la separabilidad de clases y en las métricas por clase?  
4) ¿Qué **conjunto de features** produce la mejor separación entre actividades?

---

## Tipo de problema
- **Clasificación supervisada multiclase** sobre **secuencias temporales** de landmarks (5 clases).

---

## Metodología (CRISP-DM)
- **Comprensión del problema:** detección y clasificación de actividades humanas desde landmarks de MediaPipe Pose.  
- **Comprensión de los datos:** verificación de shape, conteos por clase/velocidad, revisión de *visibility*.  
- **Preparación de datos:** renombrado de landmarks críticos, normalizaciones en Y, generación de rasgos dinámicos (velocity_x/y, **velocity_magnitude**, **acceleration**).  
- **Modelado (siguiente entrega):** SVM-RBF, Random Forest, XGBoost;
- **Evaluación:** métricas de clasificación y análisis de errores por clase.  
- **Despliegue/Iteración:** no aplica aún;

---

## Métricas para medir el progreso
- **F1-macro** (objetivo de referencia: **≥ 0.85**, ajustable tras primera línea base).  
- **F1 por clase** y **matriz de confusión** (para identificar pares confundibles como sentarse/levantarse).  
- **Precisión (Precision)** y **Cobertura (Recall)** por clase.  
- **Análisis estratificado por velocidad** (lento/rápido) y por clip (para evitar fuga).  
- **Estabilidad** entre particiones (STD de F1 en validación).

---

## Datos recolectados
- **Clips**: 10 (con 94–164 frames por clip en el top 5).  
- **Total de frames**: 981.  
- **Clases**: 5 (ver distribución arriba).  
- **Velocidad**: dos categorías (lento/rápido).  
- **Landmarks**: 33 disponibles; se usaron **11 críticos** para el análisis y generación de rasgos dinámicos.

---

## Análisis exploratorio de datos (EDA)
- **Conteos / balance**: clases relativamente próximas; **velocidad** está desbalanceada (lento>rápido).  
- **Series Y por landmark**: patrones consistentes en **caderas** por clase; mayor variabilidad en **piernas**.  
- **Rasgos dinámicos**: picos de **velocity_magnitude** distinguen momentos clave por clase (e.g., transiciones en sentarse/levantarse).  
- **Correlación (Y)**:  
  - Muy alta en pares **rodilla-rodilla** y **rodilla-tobillo** (≈0.90–0.99) → posible **redundancia** distal.  
  - Menor entre **caderas** y piernas (≈0.27–0.49) → el **core** aporta señal diferenciada útil para el clasificador.  
- **Quality/visibility**: **caderas** con valores cercanos a 1; **rodillas/tobillos** más variables. Implicación: usar **ponderación por *visibility*** y umbrales de descarte/interpolación donde haga sentido.

---

## Estrategias para conseguir más datos
1) **Recolección adicional de variariando las condiciones** (ángulo de cámara, distancia, fondo).  
2) **Balancear “rápido”**: grabar más clips rápidos para reducir asimetría lento/rápido.  
3) **Diversidad de vestimenta** distinto tipo de pantalones, uso de chaquetas o botas.

---

## Aspectos éticos (IA en este contexto)
- **Privacidad y consentimiento**: autorización explícita de participantes; uso académico; posibilidad de revocar.  
- **Minimización de datos**: almacenar **solo landmarks y metadatos necesarios**, no video crudo cuando sea posible.   
- **Transparencia**: explicar propósito y límites.  
- **Seguridad**: cifrado en repositorio privado; control de acceso y borrado responsable.

---

## Siguientes pasos para semana 14

1) **Estrategia para nuevos datos**
   - Recolectar más clips **variando condiciones controladas** (ropa, distancia, fondo).
   - **Balancear la categoría “rápido”**: asegurar que cada clase tenga ejemplos en **lento y rápido**.

2) **Preparación de datos**
   - Curación: verificar integridad de clips y **unificar FPS**.
   - Normalizaciones consistentes y centrado.
   - Construir **tres vistas de features** alineadas con las preguntas de interés:  
     a) **Todo el cuerpo** (landmarks completos)
     b) **Caderas+tronco+hombros** (core hip/torso)
     c) **Caderas+rodillas+tobillos** (miembro inferior)
   - Mantener particiones reproducibles **train/val/test** estratificadas por **clip**.

3) **Entrenamiento de modelos y ajuste de hiperparámetros**
   - Modelos: **SVM-RBF**, **Random Forest**, **XGBoost**.
   - Búsqueda (Grid/Random Search): SVM, RF, XGB
   - Validación: **k-fold por clip** y **reporte estratificado por velocidad** (lento/rápido).
   - Selección por **F1-macro** y **estabilidad** entre folds.

4) **Resultados a reportar (métricas y gráficas)**
   - **F1-macro**, **F1 por clase**, **matriz de confusión** (en test).
   - Comparativo de las **tres vistas de features** (todo vs. caderas+tronco+hombros vs. caderas+rodillas+tobillos).
   - **Importancia de features** (*permutation importance* o similar) y ejemplos de **trayectorias/velocidades** representativas.
   - Análisis de errores: pares más confundidos y breve interpretación.

5) **Plan de despliegue (MVP)**
   - Pipeline reproducible (notebook/script) que reciba un clip y devuelva predicción por ventana + visualización de skeleton/indicadores.

