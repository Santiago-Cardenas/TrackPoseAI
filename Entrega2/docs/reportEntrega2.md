# 📝 **REPORTE ENTREGA 2**

---

## **1. 📊 Resumen Ejecutivo**

### ¿Qué hicimos?
Construimos un sistema que **clasifica automáticamente 5 actividades humanas** (caminar adelante/atrás, girar, sentarse, levantarse) usando solo una cámara y Machine Learning.

### Datos recolectados
- **Entrega 1:** 981 frames, 10 clips 😅
- **Entrega 2:** 4,399 frames, 60 clips 🎉
- **Mejoras:** Grabamos con 3 ángulos, 2 distancias, 2 velocidades

### El ganador 🏆
- **Modelo:** Random Forest
- **Features:** Solo 40 (nariz, hombros, caderas) - ¡no necesitamos todo el cuerpo!
- **F1-macro:** 0.88 (88%) ✅
- **Accuracy:** 0.89 (89%) ✅

### Lo más interesante
- ✅ **Menos es más:** Usar solo el torso funcionó MEJOR que todo el cuerpo
- ✅ **Girar es fácil:** 98% de precisión
- ⚠️ **Sentarse vs Levantarse:** El modelo se confunde (son muy parecidos)
- 📉 **Velocidad importa:** Funciona casi perfecto en lento (99.6%), pero baja en rápido (64.7%)

---

## **2. 🎬 Estrategia de Recolección de Datos**

### ¿Por qué más datos?
En la Entrega 1 teníamos problemas:
- Pocos videos (solo 10)
- Ropa suelta que tapaba rodillas y tobillos 👖
- Un solo ángulo de cámara 📹
- Baja confianza en detección de piernas

### La nueva estrategia 💡

Grabamos **60 videos** variando 3 dimensiones:

#### 📐 **Ángulos de cámara (3)**
- **Derecho:** Cámara a 45° derecha
- **Izquierdo:** Cámara a 45° izquierda  
- **Centro:** Cámara frontal

*¿Por qué?* En la vida real la cámara no siempre está perfecta.

#### 📏 **Distancias (2)**
- **Cerca:** 1.5-2 metros
- **Lejos:** 3-4 metros

*¿Por qué?* Habitaciones pequeñas vs grandes.

#### ⚡ **Velocidades (2)**
- **Lento:** Movimientos pausados
- **Rápido:** Velocidad natural/rápida

*¿Por qué?* Las personas no siempre se mueven igual.

### Mejoras de calidad ✨
- ✅ Ropa ajustada (sin obstrucciones)
- ✅ Buena iluminación
- ✅ Fondo neutro
- ✅ Videos de duración similar

### Nomenclatura de archivos
```
Adelante_Lento_Der_Cerca.mp4
  ↑       ↑      ↑     ↑
Acción  Veloc  Ang  Dist
```

### Resultados 📈

| Métrica | Antes | Ahora | Mejora |
|---------|-------|-------|--------|
| Clips | 10 | 60 | 6× 🚀 |
| Frames | 981 | 4,399 | 4.5× 🚀 |
| Ángulos | 1 | 3 | 3× |
| Confidence | ~0.85 | ~0.99 | ⬆️ |

**Distribución por actividad:**
- Girar: 1,467 frames (la más larga)
- Sentarse: 843 frames
- Caminar atrás: 775 frames
- Levantarse: 670 frames
- Caminar adelante: 644 frames

---

## **3. 🔧 Preparación de Datos**

### Pipeline en 6 pasos

#### 1️⃣ **Extracción de landmarks**
- MediaPipe Pose detecta 33 puntos del cuerpo
- Cada punto tiene: x, y, z, confidence
- Total: 136 columnas base

#### 2️⃣ **Renombrar lo importante**
Convertimos `coord_x_23` → `left_hip_x` (más legible 😊)

#### 3️⃣ **Normalización**
Problema: La persona se ve más pequeña cuando está lejos
Solución: Normalizamos las coordenadas Y a rango [0, 1]

#### 4️⃣ **Feature Engineering** 🧪
Agregamos información de **movimiento**:
- **Velocidad:** ¿Qué tan rápido se mueve cada landmark?
- **Magnitud:** Velocidad total (combinando X e Y)
- **Aceleración:** ¿Está acelerando o frenando?

**¿Por qué?** Las actividades se distinguen por **cómo se mueven**, no solo por la posición.

**Resultado:** De 136 a **182 columnas**

#### 5️⃣ **Crear 3 vistas de features**

Para responder: *"¿Necesitamos todo el cuerpo o bastan algunas partes?"*

| Vista | ¿Qué incluye? | Features | 
|-------|--------------|----------|
| **A (Todo)** 🎯 | Todos los 33 landmarks | 176 |
| **B (Core)** 💪 | Nariz, hombros, caderas | 40 |
| **C (Piernas)** 🦵 | Caderas, rodillas, tobillos | 48 |

#### 6️⃣ **Split de datos**

⚠️ **Importante:** Dividimos por CLIPS, no por frames (evita data leakage)

- **Train:** 42 clips (3,176 frames) - Para entrenar
- **Val:** 6 clips (489 frames) - Para ajustar hiperparámetros
- **Test:** 12 clips (734 frames) - Para evaluar el resultado final

---

## **4. 🤖 Entrenamiento de Modelos**

### Los 3 modelos elegidos

#### **SVM (Support Vector Machine)**
- 🎯 Bueno para datos con muchas dimensiones
- Busca la mejor "línea" que separa las clases

#### **Random Forest** 🌲
- Crea muchos árboles de decisión y vota
- Robusto y fácil de interpretar

#### **XGBoost** ⚡
- El más potente (gana competencias de Kaggle)
- Crea árboles secuencialmente corrigiendo errores

### Grid Search 🔍

Probamos diferentes combinaciones de hiperparámetros:

**SVM:** 9 combinaciones
- C: [0.1, 1, 10]
- gamma: ['scale', 0.001, 0.01]

**Random Forest:** 12 combinaciones  
- n_estimators: [100, 200]
- max_depth: [10, 20, None]
- min_samples_split: [2, 5]

**XGBoost:** 12 combinaciones
- n_estimators: [100, 200]
- max_depth: [3, 5, 7]
- learning_rate: [0.01, 0.1]

### Entrenamiento

Entrenamos **9 modelos en total:**
- 3 algoritmos × 3 vistas = 9 combinaciones
- Validación cruzada (cv=3) en cada uno
- Tiempo total: ~13 minutos ⏱️

---

## **5. 📊 Resultados y Análisis**

### Tabla comparativa de los 9 modelos

| Vista | Modelo | F1-Test | Accuracy | Tiempo |
|-------|--------|---------|----------|--------|
| **B (Core)** 🏆 | **RF** | **0.88** | **0.89** | 2.6 min |
| A (Todo) | RF | 0.88 | 0.89 | 7.6 min |
| B (Core) | SVM | 0.86 | 0.87 | - |
| B (Core) | XGB | 0.84 | 0.85 | - |
| A (Todo) | XGB | 0.83 | 0.85 | - |
| A (Todo) | SVM | 0.83 | 0.85 | - |
| C (Piernas) | RF | 0.81 | 0.83 | 2.9 min |
| C (Piernas) | SVM | 0.76 | 0.79 | - |
| C (Piernas) | XGB | 0.70 | 0.73 | - |

### 🏆 El ganador: Random Forest + Vista B

**¿Por qué es sorprendente?**
- Vista B usa **solo 40 features** (vs 176 de Vista A)
- ¡Menos información dio MEJOR resultado!
- Más simple = más rápido = más interpretable

### Desempeño por clase 📈

| Actividad | F1-Score | ¿Cómo le fue? |
|-----------|----------|---------------|
| **Girar** 🔄 | 0.98 | ¡Casi perfecto! |
| **Caminar atrás** ⬅️ | 0.95 | Excelente |
| **Caminar adelante** ➡️ | 0.95 | Excelente |
| **Levantarse** ⬆️ | 0.77 | Regular |
| **Sentarse** ⬇️ | 0.76 | Regular |

### Errores más comunes 🔍

**El problema principal:**
- 43 frames de "levantarse" confundidos con "sentarse" (30% 😬)
- Son movimientos **inversos** → difíciles de distinguir en frames individuales

**¿Por qué pasa?**
- La postura intermedia es idéntica
- Solo mirando una **secuencia de frames** se puede saber si sube o baja

### Análisis por velocidad ⚡

**¡Descubrimiento importante!**

| Velocidad | F1-Score | Interpretación |
|-----------|----------|----------------|
| **Lento** 🐌 | **0.996** | ¡Casi perfecto! 🎉 |
| **Rápido** 🏃 | **0.647** | Mejorable 😅 |

**¿Qué significa?**
- El modelo funciona **EXCELENTE** para movimientos lentos
- Movimientos rápidos son más difíciles (transiciones bruscas)
- Especialmente difícil: sentarse/levantarse rápido

### Features más importantes 🔑

Top 5 que más pesan en la decisión:
1. `nose_confidence` (7.7%) - Qué tan visible está la cara
2. `nose_velocity_x` (5.0%) - Velocidad horizontal de la nariz
3. `right_hip_z` (4.9%) - Profundidad de cadera
4. `left_hip_y` (4.7%) - Altura de cadera izquierda
5. `left_hip_z` (4.7%) - Profundidad de cadera izquierda

**Patrón:** Las **caderas** y sus **velocidades** son clave 💪

---

## **6. 🌍 Análisis de Impactos**

### Aplicaciones potenciales

🏥 **Salud:**
- Monitoreo de adultos mayores
- Rehabilitación física
- Detección de caídas

🎮 **Entretenimiento:**
- Control por gestos
- Videojuegos sin controlador

⚽ **Deportes:**
- Análisis de técnica
- Entrenamiento personalizado

### Impactos Positivos ✅

1. **Accesibilidad**
   - Solo necesita una cámara (no sensores caros)
   - Funciona en celulares/computadoras normales

2. **Automatización**
   - No requiere supervisión humana constante
   - Escalable a muchos usuarios

3. **Objetividad**
   - Análisis consistente (no depende de percepción humana)

### Impactos Negativos ⚠️

1. **Privacidad**
   - Necesita grabar video de personas
   - Riesgo de mal uso de datos
   - Requiere consentimiento explícito

2. **Sesgos del modelo**
   - Entrenado con una sola persona
   - Puede no funcionar bien con:
     - Diferentes tipos de cuerpo
     - Diferentes edades
     - Personas con movilidad reducida

3. **Falsos negativos críticos**
   - 30% de error en sentarse/levantarse
   - Podría fallar detectando caídas si se parecen a "sentarse"
   - Riesgo en aplicaciones de seguridad

4. **Limitaciones técnicas**
   - Necesita buena iluminación
   - No funciona con ropa que tapa articulaciones
   - Baja precisión en movimientos rápidos

### Consideraciones Éticas 🤝

**Privacidad:**
- ✅ Procesar video localmente (no enviarlo a la nube)
- ✅ Borrar video después de extraer landmarks
- ✅ Guardar solo coordenadas anónimas

**Consentimiento:**
- ✅ Explicar claramente qué datos se capturan
- ✅ Derecho a decir que no
- ✅ Transparencia total

**Equidad:**
- ⚠️ Probar con diversidad de personas (edades, cuerpos, movilidades)
- ⚠️ No discriminar algorítmicamente
- ⚠️ No reemplazar diagnóstico profesional

### Mitigaciones propuestas 🛡️

1. **Para privacidad:**
   - Solo landmarks, no video completo
   - Encriptación de datos
   - Control de acceso estricto

2. **Para robustez:**
   - Recolectar datos de más personas
   - Más ejemplos de movimientos rápidos
   - Modelo temporal para sentarse/levantarse

3. **Para transparencia:**
   - Documentar limitaciones claramente
   - Mostrar nivel de confianza de cada predicción
   - Interfaz que indique cuándo el modelo está inseguro

---

## **8. 🎯 Conclusiones**

### Lo que logramos ✅

1. **Sistema funcional** que clasifica 5 actividades con 88% de precisión
2. **Descubrimiento clave:** Menos features (solo torso) = mejor resultado
3. **Dataset robusto:** 4,399 frames con variación de ángulos, distancias y velocidades
4. **Pipeline reproducible:** Desde video hasta predicción
5. **Análisis completo:** Sabemos dónde funciona bien y dónde no

### Hallazgos importantes 💡

#### 🏆 **Éxitos**
- Girar, caminar adelante y caminar atrás: >95% de precisión
- Movimientos lentos: casi perfectos (99.6%)
- Vista B (core) es suficiente y más eficiente

#### ⚠️ **Desafíos**
- Confusión entre sentarse/levantarse (30% error)
- Movimientos rápidos necesitan mejora (64.7% F1)
- Modelo entrenado solo con una persona

### Lecciones aprendidas 📚

1. **Más datos ≠ mejor siempre**
   - 176 features (todo el cuerpo) fue peor que 40 features (solo torso)
   - Más información puede ser ruido

2. **El contexto temporal importa**
   - Frames individuales no capturan dirección del movimiento
   - Sentarse vs levantarse necesita ver secuencias

3. **La velocidad cambia todo**
   - El mismo modelo funciona MUY diferente en lento vs rápido
   - Importante considerar para aplicaciones reales

4. **Random Forest > Otros**
   - Ganó en las 3 vistas
   - Más interpretable que XGBoost
   - Más robusto que SVM

### Trabajo futuro 🔮

#### **Mejoras técnicas:**
1. **Modelo temporal** (LSTM, Transformer) para capturar secuencias
2. **Más datos de personas diversas** (edades, tipos de cuerpo)
3. **Augmentación de datos** para movimientos rápidos
4. **Features adicionales:** dirección de movimiento, ángulos entre articulaciones

#### **Validación:**
1. Probar con usuarios diferentes
2. Evaluar en condiciones reales (no controladas)
3. Estudios de usabilidad con público objetivo

#### **Deployment**
