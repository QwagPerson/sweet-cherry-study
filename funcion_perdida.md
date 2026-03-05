### Función de pérdida final para aprender una versión suave del modelo Utah (RBF)

Esta sección define formalmente la función objetivo utilizada para estimar una función continua de aporte de frío por hora (f_\theta(T)), aprendida desde datos observacionales de temperatura y fechas fenológicas, bajo la hipótesis de que el requerimiento de frío es aproximadamente constante para cada combinación **variedad × tratamiento** a través de distintas zonas térmicas.

---

## 1. Datos, ventana de acumulación y agrupación biológica

Para cada unidad experimental (i) (p.ej., un árbol o unidad de muestreo) se dispone de:

* **Variedad** (v(i))
* **Tratamiento** (r(i)) (tratamiento de entrada a latencia)
* **Zona** (z(i)) (determina el régimen térmico al que está expuesta la unidad)
* **Fecha de inicio de acumulación de frío** ($t^{\text{in}}_i$) (p.ej., `caida_de_hojas_50_date`)
* **Fecha de término de acumulación de frío** (t^{\text{out}}_i) (p.ej., `brotacion_en_camara_50_date`)
* **Serie horaria de temperatura** (T_i(t)) para (t \in [t^{\text{in}}_i,\ t^{\text{out}}_i])

Se define el **grupo biológico**:

[
g(i) = (v(i),\ r(i))
]

y se denota por (\mathcal{I}_g) el conjunto de índices (i) que pertenecen al grupo (g).

---

## 2. Modelo continuo del aporte de frío por hora (f_\theta(T)) (RBF gaussianas)

Se modela el aporte de frío por hora como una combinación lineal de funciones base RBF gaussianas:

[
f_\theta(T)=\sum_{k=1}^{K} w_k ,\phi_k(T) + b
\quad\text{con}\quad
\phi_k(T)=\exp\left(-\frac{(T-c_k)^2}{2\sigma^2}\right)
]

donde:

* (K) es el número de bases RBF
* (c_k) son los centros (fijados) distribuidos en un rango de temperatura ([T_{\min}, T_{\max}])
* (\sigma) es el ancho de las RBF (fijo)
* (\theta = {w_1,\dots,w_K,b}) son parámetros entrenables

**Nota:** La representación RBF garantiza que (f_\theta(T)) sea una función **continua** respecto a (T).

---

## 3. Frío acumulado por unidad experimental

El frío acumulado asociado a la unidad (i) se define como la suma de los aportes horarios a lo largo de la ventana fenológica observada:

[
C_i(\theta)=\sum_{t=t^{\text{in}}_i}^{t^{\text{out}}*i} f*\theta\big(T_i(t)\big)
]

donde la suma recorre las horas contenidas en el intervalo ([t^{\text{in}}_i, t^{\text{out}}_i]).

---

## 4. Término principal: invariancia del requerimiento dentro de variedad × tratamiento

### 4.1 Hipótesis operacional

Para cada grupo (g=(\text{variedad},\ \text{tratamiento})), el requerimiento de frío es aproximadamente constante, por lo que el acumulado (C_i(\theta)) debería presentar **baja dispersión** dentro del grupo, aun cuando las unidades provengan de zonas distintas.

### 4.2 Pérdida de varianza intra-grupo

Se implementa minimizando la varianza intra-grupo del acumulado:

[
L_{\text{var}}(\theta)=\sum_{g} \mathrm{Var}_{i\in \mathcal{I}_g}\left[C_i(\theta)\right]
]

En práctica, la varianza por grupo se computa como:

[
\mathrm{Var}_{i\in \mathcal{I}_g}[C_i]
======================================

\frac{1}{|\mathcal{I}*g|}
\sum*{i\in \mathcal{I}_g}\left(C_i-\overline{C}_g\right)^2,
\qquad
\overline{C}_g=\frac{1}{|\mathcal{I}*g|}\sum*{i\in \mathcal{I}_g} C_i
]

---

## 5. Ancla al modelo Utah (prior de forma y escala relativa)

El problema de minimizar únicamente (L_{\text{var}}) es subdeterminado y admite soluciones degeneradas (por ejemplo (f_\theta(T)\equiv 0)). Para asegurar identificabilidad e interpretabilidad agronómica, se incorpora un **prior** que mantiene (f_\theta) cercano al modelo Utah clásico (f_{\text{Utah}}(T)), permitiendo un reescalamiento global aprendible (\alpha).

### 5.1 Pérdida de referencia

En una grilla de temperaturas (\mathcal{G}={T_1,\dots,T_M}) se define:

[
L_{\text{ref}}(\theta,\alpha)=
\frac{1}{M}\sum_{T\in \mathcal{G}}
\left(f_\theta(T)-\alpha f_{\text{Utah}}(T)\right)^2
]

donde:

* (f_{\text{Utah}}(T)) es el aporte por hora del modelo Utah definido por tramos (piecewise).
* (\alpha) es un escalar entrenable que permite recalibrar la escala global del Utah.

### 5.2 Restricción suave sobre (\alpha) (anti-colapso del prior)

Para evitar que el modelo reduzca (\alpha) en exceso (apagando el prior), se incorpora un castigo tipo bisagra:

[
L_{\alpha}(\alpha)=\mathrm{ReLU}(0.5-\alpha)^2
]

Esto impone que, durante el ajuste, (\alpha) se mantenga típicamente en ([0.5, \infty)), salvo evidencia fuerte en contra (controlada por el peso (\lambda_\alpha)).

---

## 6. Regularización de suavidad (curvatura)

Aunque (f_\theta) es continua por construcción (RBF), es posible obtener oscilaciones finas si (K) es grande o si los datos son ruidosos. Para favorecer soluciones suaves y fisiológicamente plausibles, se penaliza la curvatura mediante la segunda diferencia discreta en la grilla (\mathcal{G}):

[
L_{\text{smooth}}(\theta)=
\frac{1}{M-2}\sum_{j=2}^{M-1}
\left(
f_\theta(T_{j-1}) - 2 f_\theta(T_j) + f_\theta(T_{j+1})
\right)^2
]

Este término aproxima la penalización (\int (f''(T))^2 dT), reduciendo cambios bruscos de curvatura.

---

## 7. Regularizaciones guía: frío efectivo y penalización en calor

Se añaden términos suaves para guiar el aprendizaje hacia un comportamiento consistente con la interpretación usual de los modelos de frío:

* En un rango de **frío efectivo** (\mathcal{T}*{cold}) se incentiva (f*\theta(T)\ge 0).
* En un rango **cálido** (\mathcal{T}*{warm}) se incentiva (f*\theta(T)\le 0) o, equivalentemente, se penalizan aportes positivos.

Definiciones:

[
L_{\text{cold}}(\theta)=
\mathbb{E}*{T\in \mathcal{T}*{cold}}
\left[\mathrm{ReLU}\big(-f_\theta(T)\big)^2\right]
]

[
L_{\text{warm}}(\theta)=
\mathbb{E}*{T\in \mathcal{T}*{warm}}
\left[\mathrm{ReLU}\big(f_\theta(T)\big)^2\right]
]

Estos términos actúan como regularizaciones adicionales; su peso puede reducirse o anularse en experimentos de ablation.

---

## 8. Función de pérdida total

La función de pérdida final queda definida como:

$[
\boxed{
\mathcal{L}(\theta,\alpha)=
L_{\text{var}}(\theta)
+\lambda_u L_{\text{ref}}(\theta,\alpha)
+\lambda_s L_{\text{smooth}}(\theta)
+\lambda_c L_{\text{cold}}(\theta)
+\lambda_w L_{\text{warm}}(\theta)
+\lambda_\alpha L_{\alpha}(\alpha)
}
]$

donde (\lambda_u,\lambda_s,\lambda_c,\lambda_w,\lambda_\alpha \ge 0) controlan la contribución relativa de cada término.

---

## 9. Interpretación de cada componente (una línea)

* $**(L_{\text{var}})**$: fuerza la invariancia del requerimiento de frío dentro de cada **variedad×tratamiento**.
* $**(L_{\text{ref}})**$: ancla la forma del modelo a Utah (con recalibración global (\alpha)).
* $**(L_{\alpha})**$: evita que (\alpha) colapse por debajo de 0.5 (mantiene activo el ancla Utah).
*$ **(L_{\text{smooth}})**$: evita oscilaciones finas; promueve curvas fisiológicamente suaves.
* $**(L_{\text{cold}})**$: guía a aportes no negativos en frío efectivo.
* $**(L_{\text{warm}})**$: guía a aportes no positivos en calor.

---

Si quieres, también te la dejo en una versión “mini” de 8–10 líneas (solo ecuaciones + definiciones mínimas) para ponerla como recuadro en el informe.
