# <font color='#1d4ed8'>Lecture Script: Masterclass - Derivatives, Gradients and Gradient Descent</font>

---

## <font color='blue'>Agenda</font>

The lecture runs as four topics. Each topic ends with a quiz.

| Topic | What we do                           | Learning objectives covered                                                                                                                      |    Minutes    |
| :---: | :----------------------------------- | :----------------------------------------------------------------------------------------------------------------------------------------------- | :-----------: |
|   1   | Read the slope of the cost curve     | Compute the derivative of a polynomial function using the power rule<br>Apply the chain rule to differentiate a composite function                   |      30      |
|   2   | Follow the gradient uphill           | Construct the gradient vector of a multivariable function at a stated point<br>Interpret the gradient as the direction of steepest ascent at a point |      25      |
|   3   | Measure how the forecast line misses | Compute the gradient of a squared-error loss with respect to its parameters                                                                      |      25      |
|   4   | Walk downhill to the lowest cost     | Implement gradient descent to minimise a single-variable function                                                                                |      25      |
|  —  | Closing: Outcome & Summary           |                                                                                                                                                  |      15      |
|      | **Total**                      |                                                                                                                                                  | **120** |

> <font color="red">**🎯 Instructor Note**</font> Pace: Topic 1 holds two short rules, so give the chain rule the larger share. Topics 3 and 4 are where learners write code, so protect their time.
> Topic 2 has no data file; keep it moving and let the chart do the explaining.

---

## <font color='blue'>Scenario</font>

A Tata Power-style electricity distribution company buys extra power every summer evening, on top of its fixed contract, to cover a city's peak demand. The control room must decide how much to buy.

**The risk:** if the control room buys too little, the grid pays heavy shortfall charges or cuts supply to homes. If it buys too much, the company pays for power nobody uses.

> <font color="red">**🎯 Instructor Note**</font> Tata Power is only the setting. Do not claim it uses these numbers; the cost curve and the 120 evenings in `data/feeder_load.csv` are invented for teaching.
> Learners need Python 3 with NumPy 1.24 or later (Google Colab has it). The CSV has a 24 °C and a 44 °C evening, repeated temperatures, a few festival-evening spikes, and four readings capped at 88 MW.

---

## <font color='blue'>Problem Statement</font>

Meera works in the control room. Every evening she picks the extra purchase size $x$, counted in units of 10 MW, and the cost is shown in Rs lakh. Her spreadsheet had four sizes in it: 10, 30, 50 and 70 MW. They cost 30, 32, 130 and 372 lakh, so she bought 10 MW at 30 lakh and moved on.

The cheapest size was never on her sheet. It sat between her tries, at 20 MW and 22 lakh. Meera overpaid 8 lakh an evening, and over a 90-evening summer that is Rs 7.2 crore. She had also set her demand-forecast line by eye, in the same way.

> <font color='purple'>**How can Meera find the cheapest purchase size, and the best forecast line, without trying sizes one by one?**</font>

> <font color="red">**🎯 Instructor Note**</font> Poll: "At 50 MW the cost is rising as she buys more. Should she move to a bigger or a smaller purchase?" Give learners 30 seconds to respond in chat. Reveal the answer in the Closing.

---

## <font color='blue'>What We Will Use</font>

Everything the lecture needs, in one place. Each topic uses these ideas without repeating them. All the code uses NumPy only.

| Term                                                            | Meaning                                                                            | In our control room                                     |
| :-------------------------------------------------------------- | :--------------------------------------------------------------------------------- | :------------------------------------------------------ |
| **Derivative** $f'(x)$                                  | The slope of $f$ at $x$: how fast $f$ changes for one extra unit of $x$    | $C'(x)$, in lakh per 10 MW                            |
| **Power rule**                                            | Differentiate $a x^{n}$ term by term                                             | the cost curve $C(x)$                                 |
| **Chain rule**                                            | Differentiate a function whose input is another function                           | cost through the forecast, $C(x(T))$                  |
| **Partial derivative** $\dfrac{\partial F}{\partial x}$ | The slope along one variable, with the others held fixed                           | purchase $x$ or battery use $y$                     |
| **Gradient** $\nabla F$                                 | The vector of all partial derivatives at a point                                   | $[\,\partial F/\partial x,\ \partial F/\partial y\,]$ |
| **Steepest ascent**                                       | The direction in which $F$ rises fastest; its rate is the length of $\nabla F$ | the way cost climbs fastest                             |
| **Squared-error loss** $L$                              | The mean of the squared misses of a line                                           | forecast line $\hat{y}=wx+b$                          |
| **Learning rate** $\eta$                                | The size of each downhill move                                                     | a small number such as $0.02$                         |
| **Gradient descent**                                      | Repeat: move against the slope                                                     | find the cheapest purchase                              |

$$
\boxed{\frac{d}{dx}\left(a\,x^{n}\right)=a\,n\,x^{\,n-1}} \qquad \boxed{\frac{d}{dT}\,C\big(x(T)\big)=C'\big(x(T)\big)\cdot x'(T)}
$$

$$
\boxed{\nabla F=\left[\frac{\partial F}{\partial x},\ \frac{\partial F}{\partial y}\right]} \qquad \boxed{x_{\text{new}}=x-\eta\,f'(x)}
$$

> <font color='#b45309'>**Watch out for**</font>
>
> - A constant has slope $0$; $x^{1}$ has slope $1$.
> - The chain rule has two factors. Leaving out the inner slope is the usual slip.
> - The gradient depends on the point. A new point gives a new vector.
> - Descent moves against the slope, so the sign in $x-\eta f'(x)$ is a minus.

---

## <font color='blue'>How: Walking Through the Control Room</font>

```mermaid
flowchart LR
    A["Topic 1<br>Read the slope"] --> B["Topic 2<br>Follow the gradient"]
    B --> C["Topic 3<br>Measure the forecast miss"]
    C --> D["Topic 4<br>Walk downhill"]
    D --> E["Closing<br>Which size to buy?"]
    style A fill:#dbeafe,stroke:#2563eb
    style B fill:#dcfce7,stroke:#16a34a
    style C fill:#fef3c7,stroke:#d97706
    style D fill:#ede9fe,stroke:#7c3aed
    style E fill:#fee2e2,stroke:#dc2626
```

---

## <font color='blue'>Topic 1: Read the Slope of the Cost Curve</font> `30 min`

> <font color='purple'>**Meera's sheet held four costs. What can the slope of the cost curve tell her that four tries could not?**</font>

On a typical summer evening, the cost of buying $x$ units (of 10 MW) of extra power is the polynomial

$$
C(x)=x^{3}+3x^{2}-24x+50 \quad \text{(Rs lakh)}
$$

### <font color='green'>Example: the slope of the cost curve by the power rule</font>

Differentiate each term with the power rule, and add the results. The constant $50$ has slope $0$.

$$
\boxed{\frac{d}{dx}\left(a\,x^{n}\right)=a\,n\,x^{\,n-1}}
$$

$$
C'(x)=3x^{2}+6x-24 \qquad\Rightarrow\qquad C'(5)=3(25)+6(5)-24=81
$$

**Predict before running: What will happen?**

> <font color="red">**🎯 Instructor Note**</font> Expected: the new coefficients $3, 6, -24$ with powers $2, 1, 0$, and a slope of $81$. A common wrong guess is $130$, which is the cost $C(5)$ and not its slope.

```python
import numpy as np

coef  = np.array([1, 3, -24])   # terms x^3, x^2, x (the constant 50 has slope 0)
power = np.array([3, 2, 1])
d_coef, d_power = coef * power, power - 1
print("coefficients:", d_coef)
print("powers      :", d_power)
x = 5.0
print("C'(5) =", np.sum(d_coef * x ** d_power))
```

**Output**

```
coefficients: [  3   6 -24]
powers      : [2 1 0]
C'(5) = 81.0
```

#### <font color='#c2410c'>Code Explanation</font>

- **Line 1** loads NumPy.
- **Lines 3 and 4** list the terms $x^{3}, x^{2}, x$ by their coefficient and their power. The constant is left out because its slope is $0$.
- **Line 5** is the rule itself: multiply each coefficient by its power, and lower each power by $1$.
- **Lines 6 and 7** print the new coefficients and powers. They read $3x^{2}+6x^{1}-24x^{0}$, which is $C'(x)$.
- **Lines 8 and 9** put $x=5$ into $C'(x)$ and add the three terms.

> <font color='green'>**Result**</font> At $x=5$ the slope is $+81$: each extra 10 MW adds about $81$ lakh to the cost. The cost is climbing, so buying less would help.

<p align="center"><img src="https://coding-platform.s3.amazonaws.com/dev/lms/tickets/3b46edd1-1982-4b57-992a-14a564e1b54c/rkjPbnTmSidZljRl.png" alt="Left: the cost curve C(x) with tangent lines at x = 0, 2 and 5, with slopes -24, 0 and 81. Right: cost against forecast temperature with a tangent at 40 °C of slope 9.6." width="900"></p>

### <font color='green'>Extending it: the slope at three purchase sizes</font>

> <font color='purple'>**What is the slope at 0, 20 and 50 MW, and can we check it without the rule?**</font> A tiny step gives a second opinion: the cost change over a very small interval, divided by its width.

> <font color="red">**🎯 Instructor Note**</font> Ask for a prediction first. At $x=2$ learners often guess a small positive number; the slope there is exactly $0$, the bottom of the curve.

```python
def C(x):  return x**3 + 3*x**2 - 24*x + 50
def dC(x): return 3*x**2 + 6*x - 24

h = 1e-3
for x in (0.0, 2.0, 5.0):
    check = (C(x + h) - C(x - h)) / (2 * h)
    print(f"x = {x}: cost = {C(x)}, slope = {dC(x)}, tiny-step check = {check:.3f}")
```

**Output**

```
x = 0.0: cost = 50.0, slope = -24.0, tiny-step check = -24.000
x = 2.0: cost = 22.0, slope = 0.0, tiny-step check = 0.000
x = 5.0: cost = 130.0, slope = 81.0, tiny-step check = 81.000
```

#### <font color='#c2410c'>Code Explanation</font>

- **Lines 1 and 2** write the cost curve and the slope formula we found as two small functions.
- **Line 4** picks a small width $h$ for the tiny step.
- **Line 5** loops over the three purchase sizes.
- **Line 6** steps a little to the right and a little to the left of $x$, and divides the change in cost by the width $2h$.
- **Line 7** prints the cost, the slope from the power rule and the tiny-step check side by side.

> <font color='green'>**Result**</font> The slope is $-24$ at $x=0$ (cost falls as she buys more), $0$ at $x=2$ and $+81$ at $x=5$. The power rule and the tiny step agree. The slope is $0$ exactly where the cost is lowest, $22$ lakh at 20 MW.

### <font color='green'>Extending it: a function inside a function (the chain rule)</font>

> <font color='purple'>**If the control room buys $x(T)=T^{2}/400$ units for a forecast of $T$ °C, how fast does the evening cost change for each extra degree?**</font> The cost depends on $x$, and $x$ depends on $T$.

$$
\boxed{\frac{d}{dT}\,C\big(x(T)\big)=C'\big(x(T)\big)\cdot x'(T)} \qquad x'(T)=\frac{2T}{400}=\frac{T}{200}
$$

At $T=40$: $x=\dfrac{1600}{400}=4$, $C'(4)=3(16)+6(4)-24=48$, $x'(40)=0.2$, so $\dfrac{dC}{dT}=48\times0.2=9.6$.

**Predict before running: What will happen?**

> <font color="red">**🎯 Instructor Note**</font> Expected: $x=4.0$, $C'(x)=48.0$, $x'(T)=0.2$ and a product of $9.6$. A common wrong answer is $48$, which leaves out the inner slope.

```python
def x_of_T(T): return T**2 / 400
def dx_dT(T):  return T / 200

T = 40.0
x = x_of_T(T)
print("x =", x, "| C'(x) =", dC(x), "| x'(T) =", dx_dT(T))
print("dC/dT =", round(dC(x) * dx_dT(T), 4))
```

**Output**

```
x = 4.0 | C'(x) = 48.0 | x'(T) = 0.2
dC/dT = 9.6
```

#### <font color='#c2410c'>Code Explanation</font>

- **Lines 1 and 2** write the inner function $x(T)$ and its slope, found with the power rule on $T^{2}/400$.
- **Lines 4 and 5** set the forecast to $40$ °C and find the purchase $x$ the rule asks for.
- **Line 6** prints the three pieces: the purchase, the outer slope $C'(x)$ and the inner slope $x'(T)$.
- **Line 7** multiplies the outer slope by the inner slope. That product is the chain rule.

> <font color='green'>**Result**</font> At a 40 °C forecast, each extra degree adds about $9.6$ lakh to the evening cost.

```python
h = 1e-5
check = (C(x_of_T(T + h)) - C(x_of_T(T - h))) / (2 * h)
print("tiny-step check:", round(check, 4))
print("outer slope only:", dC(x))
```

**Output**

```
tiny-step check: 9.6
outer slope only: 48.0
```

#### <font color='#c2410c'>Code Explanation</font>

- **Lines 1 and 2** step the forecast $T$ a tiny bit up and down and divide the change in cost by $2h$, going through the whole chain $T\to x\to C$.
- **Line 3** prints that check, and **line 4** prints the outer slope alone, which is the wrong answer to the question.

> <font color='#b45309'>**Common mistake**</font> Using only the outer slope $C'(x)=48$ answers "how fast does cost change per unit of $x$", not "per degree". The inner slope $0.2$ turns the units from $x$ into $T$.

### <font color='green'>Extending it: a cool evening, a mild evening and a hot one</font>

> <font color='purple'>**Does a hotter forecast always make the evening cost more?**</font> Give learners 30 seconds to respond in chat first, then run it.

> <font color="red">**🎯 Instructor Note**</font> Expected: no. At 20 °C the rule asks for $x=1$, below the best size, so a hotter forecast moves it toward the bottom and the cost falls.

```python
for T in (20.0, 800 ** 0.5, 40.0):
    x = x_of_T(T)
    print(f"T = {T:5.2f}: x = {x:4.2f}, dC/dT = {dC(x) * dx_dT(T):6.3f}")
```

**Output**

```
T = 20.00: x = 1.00, dC/dT = -1.500
T = 28.28: x = 2.00, dC/dT =  0.000
T = 40.00: x = 4.00, dC/dT =  9.600
```

#### <font color='#c2410c'>Logic Explanation</font>

- At $T=20$: $x=1$ and $C'(1)=-15$, so $\dfrac{dC}{dT}=-15\times0.1=-1.5$. The cost falls as the forecast rises.
- At $T=\sqrt{800}\approx28.28$: $x=2$, the cheapest size, so $C'(x)=0$ and the product is $0$.
- At $T=40$: both factors are positive, so the cost rises by $9.6$ lakh per degree.

> <font color='green'>**Result**</font> The sign of $\dfrac{dC}{dT}$ follows the sign of $C'(x)$, because $x'(T)$ is positive for $T>0$.

### <font color='purple'>Quiz 1</font>

**Question:** Let $h(x)=(x^{2}+1)^{3}$. What is $h'(1)$?

**Options:**

- A) $2$
- B) $12$
- C) $24$
- D) $14$

**Correct Answer:** C

**Explanation:**

- **A)** Wrong: $2$ is the inner slope $2x$ alone, with the outer slope missing.
- **B)** Wrong: $12$ is the outer slope $3(x^{2}+1)^{2}=3(4)$ alone, with the inner slope missing.
- **C)** Correct: $h'(x)=3(x^{2}+1)^{2}\cdot2x$, so $h'(1)=3(4)(2)=24$.
- **D)** Wrong: $14=12+2$ adds the two slopes, but the chain rule multiplies them.

---

## <font color='blue'>Topic 2: Follow the Gradient Uphill</font> `25 min`

> <font color='purple'>**The control room can also use a battery. With two choices at once, which way does the cost climb fastest?**</font>

Let $x$ be the purchase (units of 10 MW) and $y$ the battery discharge (MWh). The evening cost, in Rs lakh, is

$$
F(x,y)=C(x)+2y^{2}-8y+xy
$$

The last term is a charge for running both at once.

### <font color='green'>Example: the gradient at a stated point</font>

> <font color='purple'>**At the plan $x=3$, $y=2$, how fast does the cost change with each of the two choices?**</font> A partial derivative treats the other variable as a fixed number.

$$
\frac{\partial F}{\partial x}=C'(x)+y=3x^{2}+6x-24+y \qquad \frac{\partial F}{\partial y}=4y-8+x
$$

$$
\boxed{\nabla F=\left[\frac{\partial F}{\partial x},\ \frac{\partial F}{\partial y}\right]} \qquad \nabla F(3,2)=\big[\,27+18-24+2,\ \ 8-8+3\,\big]=[\,23,\ 3\,]
$$

**Predict before running: What will happen?**

> <font color="red">**🎯 Instructor Note**</font> Expected: $F(p)=30.0$ and the gradient $[23, 3]$. A common wrong guess is to give both entries the same slope, because learners forget that $y$ is held fixed in $\partial F/\partial x$.

```python
def F(p):
    x, y = p
    return C(x) + 2*y**2 - 8*y + x*y
def grad_F(p):
    x, y = p
    return np.array([dC(x) + y, 4*y - 8 + x])
p = np.array([3.0, 2.0])
print("F(p)      =", F(p))
print("grad F(p) =", grad_F(p))
```

**Output**

```
F(p)      = 30.0
grad F(p) = [23.  3.]
```

#### <font color='#c2410c'>Code Explanation</font>

- **Lines 1 to 3** write the cost of a plan $p=(x,y)$. It reuses the function `C` from Topic 1.
- **Lines 4 to 6** write the two partial derivatives and return them as one vector. The first entry reuses `dC`, plus $y$.
- **Line 7** states the plan: $x=3$, $y=2$.
- **Lines 8 and 9** print the cost and the gradient at that plan.

> <font color='green'>**Result**</font> At this plan the cost is $30$ lakh and the gradient is $[23,\ 3]$. The cost is far more sensitive to the purchase than to the battery.

<p align="center"><img src="https://coding-platform.s3.amazonaws.com/dev/lms/tickets/2b35d8b7-471f-44da-8b46-a1cdcbada803/jioTjiLBbEgKlVj3.png" alt="Left: contour lines of the cost F(x, y) with the gradient arrow at the point (3, 2) pointing uphill and the opposite arrow pointing downhill. Right: the rate of change of cost for a step in each direction from 0 to 360 degrees, with a peak of about 23.2 near 7 degrees." width="900"></p>

### <font color='green'>Extending it: one variable at a time</font>

> <font color='purple'>**Does a tiny step along each axis give the same two numbers as the gradient?**</font> Move only $x$, then only $y$.

```python
h = 1e-5
for name, e in (("x", np.array([1.0, 0.0])), ("y", np.array([0.0, 1.0]))):
    slope = (F(p + h*e) - F(p - h*e)) / (2 * h)
    print("slope along", name, "=", round(slope, 3))
```

**Output**

```
slope along x = 23.0
slope along y = 3.0
```

#### <font color='#c2410c'>Code Explanation</font>

- **Line 2** loops over two unit directions: $[1,0]$ moves only $x$, and $[0,1]$ moves only $y$.
- **Line 3** takes the same tiny step as in Topic 1, but along one axis.
- **Line 4** prints each slope, rounded.

> <font color='green'>**Result**</font> The two tiny-step slopes match the two entries of the gradient.

### <font color='green'>Extending it: which direction is steepest?</font>

> <font color='purple'>**Of all the directions the plan could move in, which one raises the cost fastest?**</font> Try 360 directions, one for each degree, and take a tiny step in each.

$$
\text{rate in direction } \mathbf{u}\ \approx\ \nabla F\cdot\mathbf{u}\ =\ \lVert\nabla F\rVert\cos\theta,\qquad \max=\lVert\nabla F\rVert=\sqrt{23^{2}+3^{2}}\approx23.2
$$

**Predict before running: What will happen?**

> <font color="red">**🎯 Instructor Note**</font> Expected: a direction near $7$ degrees (almost along $x$, tilted slightly up) and a rate near $23.2$. A common wrong guess is $45$ degrees.

```python
angles = np.deg2rad(np.arange(360))
dirs = np.column_stack([np.cos(angles), np.sin(angles)])
h = 0.001
rise = np.array([F(p + h*d) - F(p) for d in dirs]) / h
g = grad_F(p)
print("steepest direction tried:", np.argmax(rise), "degrees, rate", round(rise.max(), 1))
print("gradient direction      :", round(np.degrees(np.arctan2(g[1], g[0])), 1), "degrees, length", round(np.linalg.norm(g), 1))
```

**Output**

```
steepest direction tried: 7 degrees, rate 23.2
gradient direction      : 7.4 degrees, length 23.2
```

#### <font color='#c2410c'>Code Explanation</font>

- **Lines 1 and 2** build 360 unit arrows, one per degree, as pairs $[\cos\theta,\ \sin\theta]$.
- **Lines 3 and 4** take a tiny step $h$ in each direction and divide the rise in cost by $h$, giving a rate for each of the 360 directions.
- **Line 5** reads the gradient at $p$.
- **Lines 6 and 7** print the direction with the largest rate, and the angle and length of the gradient. `arctan2` gives the angle of the vector $[23,\ 3]$.

> <font color='green'>**Result**</font> The gradient points in the direction of steepest ascent, and its length, about $23.2$, is the rate of rise in that direction.

### <font color='green'>Extending it: uphill, downhill and sideways</font>

> <font color='purple'>**What happens to the cost if we step with the gradient, against it, or at a right angle to it?**</font> Give learners 30 seconds to respond in chat first.

> <font color="red">**🎯 Instructor Note**</font> Expected: about $+23.2$ with it, about $-23.2$ against it, and about $0$ at a right angle, where the cost stays level.

```python
u = g / np.linalg.norm(g)
side = np.array([-u[1], u[0]])
for name, d in (("with gradient", u), ("against gradient", -u), ("sideways", side)):
    print(f"{name:16s}: {(F(p + h*d) - F(p)) / h:7.2f}")
```

**Output**

```
with gradient   :   23.21
against gradient:  -23.18
sideways        :    0.00
```

#### <font color='#c2410c'>Code Explanation</font>

- **Line 1** shrinks the gradient to length $1$, so it is a pure direction.
- **Line 2** turns that arrow a quarter turn, which gives a direction at a right angle to it.
- **Lines 3 and 4** take the same tiny step in each of the three directions and print the rate of rise.

> <font color='green'>**Result**</font> Cost climbs fastest along $\nabla F$, falls fastest along $-\nabla F$, and stays level at a right angle to it.

### <font color='green'>Extending it: the gradient at a different point</font>

> <font color='purple'>**Does the gradient stay the same when the plan changes?**</font> Try the plan $(1.5,\ 3)$.

```python
q = np.array([1.5, 3.0])
print("grad F(q) =", grad_F(q))
print("downhill direction:", -grad_F(q))
```

**Output**

```
grad F(q) = [-5.25  5.5 ]
downhill direction: [ 5.25 -5.5 ]
```

#### <font color='#c2410c'>Code Explanation</font>

- **Line 1** states the new plan $q$: $x=1.5$, $y=3$.
- **Line 2** reuses `grad_F` to read the gradient at $q$.
- **Line 3** flips its sign, which gives the downhill direction.

> <font color='green'>**Result**</font> At $q$ the gradient is $[-5.25,\ 5.5]$, so cost climbs fastest by buying less and discharging more. To cut cost, the plan should do the opposite. The gradient depends on the point.

### <font color='purple'>Quiz 2</font>

**Question:** Let $G(x,y)=x^{2}y+3y$. At the point $(2,\,1)$, which vector points in the direction of steepest ascent of $G$?

**Options:**

- A) $[4,\ 7]$
- B) $[-4,\ -7]$
- C) $[7,\ 4]$
- D) $[4,\ 4]$

**Correct Answer:** A

**Explanation:**

- **A)** Correct: $\dfrac{\partial G}{\partial x}=2xy=4$ and $\dfrac{\partial G}{\partial y}=x^{2}+3=7$ at $(2,1)$, so $\nabla G=[4,\ 7]$.
- **B)** Wrong: $[-4,\ -7]$ is the direction of steepest descent, the opposite way.
- **C)** Wrong: $[7,\ 4]$ swaps the two partial derivatives; the first entry belongs to $x$.
- **D)** Wrong: $[4,\ 4]$ drops the $3y$ term, so $\partial G/\partial y$ comes out as $x^{2}=4$ and not $7$.

---

## <font color='blue'>Topic 3: Measure How the Forecast Line Misses</font> `25 min`

> <font color='purple'>**Meera set her forecast line by eye. How do we measure how badly it misses, and which way to move it?**</font>

The line is $\hat{y}=w\,x+b$, where $x$ is the heat (the evening temperature minus 30 °C) and $\hat{y}$ is the predicted peak load in MW. So $b$ is the load at 30 °C, and $w$ is the extra MW for each degree above that. For $n$ evenings, the squared-error loss is the mean of the squared misses:

$$
L(w,b)=\frac{1}{n}\sum_{i=1}^{n}u_i^{2},\qquad u_i=y_i-(w\,x_i+b)
$$

Each term is a square of a bracket, so the chain rule applies: the outer slope is $2u_i$, and the inner slopes are $-x_i$ for $w$ and $-1$ for $b$.

$$
\boxed{\frac{\partial L}{\partial w}=-\frac{2}{n}\sum_{i=1}^{n}x_i\,u_i \qquad \frac{\partial L}{\partial b}=-\frac{2}{n}\sum_{i=1}^{n}u_i}
$$

### <font color='green'>Example: one evening by hand</font>

Heat $x=35-30=5$. Prediction $\hat{y}=1(5)+60=65$. Miss $u=70.5-65=5.5$. Loss $u^{2}=30.25$.

$$
\frac{\partial L_i}{\partial w}=2u\cdot(-x)=-2(5.5)(5)=-55 \qquad \frac{\partial L_i}{\partial b}=2u\cdot(-1)=-2(5.5)=-11
$$

**Predict before running: What will happen?**

> <font color="red">**🎯 Instructor Note**</font> Expected: shape $(120, 2)$, miss $5.5$, loss $30.25$ and slopes $-55$ and $-11$. Both slopes are negative: the line predicts too low.

```python
data = np.loadtxt("data/feeder_load.csv", delimiter=",", skiprows=1, usecols=(1, 2))
heat, load = data[:, 0] - 30, data[:, 1]     # degrees above 30 C, peak load in MW
print(data.shape)
w, b = 1.0, 60.0
x1, y1 = heat[3], load[3]                    # day 4
u1 = y1 - (w * x1 + b)
print("heat", x1, "load", y1, "miss", u1, "loss", u1**2)
print("dL/dw =", -2 * u1 * x1, " dL/db =", -2 * u1)
```

**Output**

```
(120, 2)
heat 5.0 load 70.5 miss 5.5 loss 30.25
dL/dw = -55.0  dL/db = -11.0
```

#### <font color='#c2410c'>Code Explanation</font>

- **Line 1** reads the two columns of the file, temperature and peak load, into one table of 120 rows.
- **Line 2** makes `heat` (degrees above 30 °C) and `load` (MW). Heat keeps the numbers small.
- **Line 3** prints the table's shape: 120 rows, 2 columns.
- **Lines 4 and 5** state the guess $w=1,\ b=60$ and pick day 4, which is row index $3$.
- **Line 6** computes the miss $u=y-(wx+b)$.
- **Lines 7 and 8** print the loss $u^{2}$ and the two slopes from the chain rule.

> <font color='green'>**Result**</font> One evening misses by $5.5$ MW. Both slopes are negative, so raising $w$ or $b$ would lower this evening's loss.

<p align="center"><img src="https://coding-platform.s3.amazonaws.com/dev/lms/tickets/d7bda219-d2bd-4fcf-8f68-4312e9a1b2eb/PeszlA1nrUA1lEPf.png" alt="Left: 120 evenings as points of heat against peak load with the guessed line w = 1, b = 60 and the best line, and day 4's miss marked. Right: the loss as a function of w with b fixed at 60, with a tangent at w = 1 of slope -69.95." width="900"></p>

### <font color='green'>Extending it: all 120 evenings</font>

> <font color='purple'>**How badly does the guessed line $w=1,\ b=60$ miss over all 120 evenings, and which way should $w$ and $b$ move?**</font> Replace the single miss with the mean over all evenings.

> <font color="red">**🎯 Instructor Note**</font> Expected: a loss near $57$ and both gradient entries negative, with $\partial L/\partial w$ much larger in size than $\partial L/\partial b$ because it is weighted by heat.

```python
def loss(w, b):
    return np.mean((load - (w * heat + b)) ** 2)
def grad_loss(w, b):
    miss = load - (w * heat + b)
    return np.array([-2 * np.mean(heat * miss), -2 * np.mean(miss)])
print("loss     :", round(loss(1.0, 60.0), 3))
print("gradient :", np.round(grad_loss(1.0, 60.0), 3))
```

**Output**

```
loss     : 56.743
gradient : [-69.952 -11.152]
```

#### <font color='#c2410c'>Code Explanation</font>

- **Lines 1 and 2** write the loss: the mean of the squared misses over all evenings, each miss $u_i$ computed at once for the whole array.
- **Lines 3 to 5** write the gradient from the boxed formulas. `heat * miss` is each evening's $x_i u_i$.
- **Lines 6 and 7** print the loss and the gradient at the same guess $w=1,\ b=60$.

> <font color='green'>**Result**</font> The loss is about $56.7$ MW squared, and the gradient is about $[-69.95,\ -11.15]$. Day 4's slopes ($-55$ and $-11$) are one evening's share of this.

### <font color='green'>Extending it: check the gradient with a tiny step</font>

> <font color='purple'>**Does a tiny step on the loss give the same two slopes as the formula?**</font> Nudge $w$, then $b$, and watch the loss.

```python
h = 1e-5
d_w = (loss(1.0 + h, 60.0) - loss(1.0 - h, 60.0)) / (2 * h)
d_b = (loss(1.0, 60.0 + h) - loss(1.0, 60.0 - h)) / (2 * h)
print("tiny-step check:", round(d_w, 3), round(d_b, 3))
```

**Output**

```
tiny-step check: -69.952 -11.152
```

#### <font color='#c2410c'>Code Explanation</font>

- **Line 1** picks a tiny width $h$.
- **Line 2** nudges only $w$ up and down, with $b$ held at $60$, and divides the change in loss by $2h$.
- **Line 3** does the same for $b$, with $w$ held at $1$.
- **Line 4** prints both slopes, rounded like the gradient above.

> <font color='green'>**Result**</font> The tiny-step slopes match the gradient from the formula.

> <font color='#b45309'>**Common mistake**</font> Dropping the minus sign. The miss is $y-\hat{y}$, so the inner slopes are $-x_i$ and $-1$. A positive gradient here would send the line the wrong way.

### <font color='green'>Extending it: the gradient at the best line</font>

> <font color='purple'>**What is the gradient of the loss at the best possible line?**</font> `np.polyfit` is NumPy's own least-squares line. We use it only to find the best $w$ and $b$ to test against.

> <font color="red">**🎯 Instructor Note**</font> Ask for a prediction first. Expected: both entries are $0$ (shown as rounded), because the bottom of the loss is flat.

```python
w_best, b_best = np.polyfit(heat, load, 1)
print("best line: w =", round(w_best, 4), " b =", round(b_best, 4))
print("loss there:", round(loss(w_best, b_best), 3))
print("gradient there:", np.round(grad_loss(w_best, b_best), 6) + 0.0)
```

**Output**

```
best line: w = 2.0216  b = 62.1705
loss there: 8.909
gradient there: [0. 0.]
```

#### <font color='#c2410c'>Code Explanation</font>

- **Line 1** asks NumPy for the slope and intercept of the best straight line through the points.
- **Lines 2 and 3** print them, and the loss at that line.
- **Line 4** prints the gradient there. Adding `0.0` only turns a printed $-0.0$ into $0.0$.

> <font color='green'>**Result**</font> The best line is about $w=2.02$ MW per degree and $b=62.17$ MW. There the loss falls from $56.7$ to $8.9$, and the gradient is $0$.

> <font color='#b45309'>**Common mistake**</font> Trusting the line outside the data. It was fitted on evenings from 24 °C to 44 °C, so a 48 °C forecast is a guess. Squaring also makes big misses count heavily, so the festival spikes and the readings capped at 88 MW pull the line more than ordinary evenings do.

### <font color='purple'>Quiz 3</font>

**Question:** For one evening with heat $x=2$ and load $y=10$, the line is $w=1,\ b=3$. The loss for this evening is $u^{2}$ with $u=y-(wx+b)$. What is $\dfrac{\partial L}{\partial w}$?

**Options:**

- A) $20$
- B) $10$
- C) $-10$
- D) $-20$

**Correct Answer:** D

**Explanation:**

- **A)** Wrong: $20$ has the right size but the wrong sign; the inner slope of $u$ with respect to $w$ is $-x$, which is negative.
- **B)** Wrong: $10=2u$ is the outer slope alone. The inner slope $-x=-2$ is missing.
- **C)** Wrong: $-10$ is $\partial L/\partial b=2u(-1)$, the slope for the intercept and not for $w$.
- **D)** Correct: $\hat{y}=2+3=5$, $u=10-5=5$, and $\dfrac{\partial L}{\partial w}=2u\cdot(-x)=2(5)(-2)=-20$.

---

## <font color='blue'>Topic 4: Walk Downhill to the Lowest Cost</font> `25 min`

> <font color='purple'>**Can Meera reach the cheapest purchase size by following the slope, one small move at a time?**</font>

The slope says which way the cost rises. To lower the cost, move the other way. Gradient descent repeats one move: subtract a small multiple of the slope. The small multiple $\eta$ is the learning rate, the size of each move.

$$
\boxed{x_{\text{new}}=x-\eta\,f'(x)}
$$

### <font color='green'>Example: walk down the cost curve from 50 MW</font>

First move by hand: $C'(5)=81$, so $x_{1}=5-0.02(81)=3.38$.

**Predict before running: What will happen?**

> <font color="red">**🎯 Instructor Note**</font> Expected: $x$ drops fast at first ($5, 3.38, 2.77, \dots$) and settles near $2$, with a cost near $22$. A common wrong guess is that it jumps straight to $2$ in one move.

```python
def descend(slope, x, lr, steps):
    path = [x]
    for _ in range(steps):
        x = x - lr * slope(x)
        path.append(x)
    return np.array(path)
path = descend(dC, 5.0, 0.02, 25)
print(np.round(path[[0, 1, 2, 3, 5, 10, 25]], 4))
print("cost at start:", C(path[0]), "| cost at end:", round(C(path[-1]), 4))
```

**Output**

```
[5.     3.38   2.7689 2.4566 2.1743 2.018  2.    ]
cost at start: 130.0 | cost at end: 22.0
```

#### <font color='#c2410c'>Code Explanation</font>

- **Line 1** defines `descend`, which takes a slope function, a start, a learning rate and a number of moves.
- **Line 2** starts a list that remembers every position.
- **Lines 3 and 4** repeat the update $x\leftarrow x-\eta f'(x)$ for the stated number of moves.
- **Line 5** saves each new position.
- **Line 6** returns the whole path as an array.
- **Line 7** runs it on `dC` from Topic 1, from $x=5$.
- **Lines 8 and 9** print the positions after moves 0, 1, 2, 3, 5, 10 and 25, then the cost at the start and the end.

> <font color='green'>**Result**</font> The walk goes from $5$ (130 lakh) to $2.0$ (22 lakh). The moves shrink on their own, because the slope shrinks as the curve flattens.

<p align="center"><img src="https://coding-platform.s3.amazonaws.com/dev/lms/tickets/e50ad37b-099d-4d08-abfd-2d4a2d58ec7c/yGYdf3QFpsfzeaFQ.png" alt="Left: the cost curve with the gradient descent path from x = 5 to x = 2 with learning rate 0.02. Right: the squared-error loss against w with the descent path from w = 1 to w = 2.04." width="900"></p>

### <font color='green'>Extending it: the size of each move</font>

> <font color='purple'>**What happens if each move is much smaller, or much bigger, than $0.02$?**</font> Give learners 30 seconds to vote in chat first: which one fails?

> <font color="red">**🎯 Instructor Note**</font> Expected: $0.001$ is far too slow, $0.07$ overshoots then settles, and $0.12$ runs away. A common wrong guess is that a bigger rate always gets there sooner.

```python
for lr in (0.001, 0.02, 0.07, 0.12):
    path = descend(dC, 5.0, lr, 5)
    print(f"lr = {lr:5.3f}: first moves {np.round(path[:3], 2)}, x after 5 moves = {path[-1]:.2f}")
```

**Output**

```
lr = 0.001: first moves [5.   4.92 4.84], x after 5 moves = 4.62
lr = 0.020: first moves [5.   3.38 2.77], x after 5 moves = 2.17
lr = 0.070: first moves [ 5.   -0.67  1.2 ], x after 5 moves = 2.01
lr = 0.120: first moves [ 5.   -4.72 -6.46], x after 5 moves = -1841.97
```

#### <font color='#c2410c'>Code Explanation</font>

- **Line 1** tries four learning rates.
- **Line 2** runs five moves from $x=5$ for each.
- **Line 3** prints the first three positions and the position after five moves.

> <font color='green'>**Result**</font> At $0.001$ each move is only about $0.08$, so it crawls. At $0.02$ it closes in on $2$. At $0.07$ the first move jumps past the bottom to $-0.67$ and then comes back. At $0.12$ the first move lands at $-4.72$, past $x=-4$ where the slope changes sign, so every later move heads further away. A negative purchase has no meaning for Meera; the formula still runs, which is why the learning rate must stay small.

<p align="center"><img src="https://coding-platform.s3.amazonaws.com/dev/lms/tickets/44c6c81b-82f4-4008-99c6-8cd4b2a42149/ruxLIRmysq1Uzudw.png" alt="Left: position x against move number for learning rates 0.001, 0.02 and 0.07 over 25 moves; the first is still above 3.5, the others settle at 2. Right: for learning rate 0.12 the position runs away from 5 to about -1842 in five moves, on a log-style axis." width="900"></p>

> <font color='#b45309'>**Common mistake**</font> Adding the slope instead of subtracting it climbs the hill. Use $x-\eta f'(x)$ to minimise.

### <font color='green'>Extending it: start on the other side</font>

> <font color='purple'>**What if Meera starts at 0 MW, below the best size?**</font> Give learners 30 seconds to respond in chat: which way will the first move go?

> <font color="red">**🎯 Instructor Note**</font> Expected: right, toward larger $x$. A negative slope is subtracted, so $x$ grows. It reaches $2$ from below.

```python
path = descend(dC, 0.0, 0.02, 25)
print(np.round(path[[0, 1, 2, 5, 10, 25]], 4))
print("slope at start:", dC(0.0), "| slope at end:", round(dC(path[-1]), 4))
```

**Output**

```
[0.     0.48   0.8886 1.6371 1.9575 1.9999]
slope at start: -24.0 | slope at end: -0.001
```

#### <font color='#c2410c'>Code Explanation</font>

- **Line 1** runs the same `descend` with the same learning rate, but starts at $x=0$.
- **Line 2** prints the positions after moves 0, 1, 2, 5, 10 and 25.
- **Line 3** prints the slope at the start and at the end of the walk.

> <font color='green'>**Result**</font> From the left, the walk climbs from $0$ to about $2$, and the slope goes from $-24$ to about $0$. Whichever side we start, the sign of the slope sends us toward the bottom.

### <font color='green'>Extending it: walk down the forecast loss</font>

> <font color='purple'>**Can the same loop fit the slope $w$ of the forecast line?**</font> With $b$ fixed, the loss depends on $w$ alone, so it is a single-variable function.

Setting the slope to zero gives the exact answer to compare with: $w=\dfrac{\sum x_i\,(y_i-62)}{\sum x_i^{2}}$.

```python
def slope_w(w):
    return grad_loss(w, 62.0)[0]
path = descend(slope_w, 1.0, 0.01, 30)
w_exact = np.sum(heat * (load - 62.0)) / np.sum(heat ** 2)
print("descent w after 30 moves:", round(path[-1], 4))
print("exact best w for b = 62 :", round(w_exact, 4))
```

**Output**

```
descent w after 30 moves: 2.0425
exact best w for b = 62 : 2.0425
```

#### <font color='#c2410c'>Code Explanation</font>

- **Lines 1 and 2** make a single-variable slope function: the $w$ entry of the gradient from Topic 3, with $b$ fixed at $62$.
- **Line 3** runs the same `descend` from $w=1$ with learning rate $0.01$.
- **Line 4** computes the exact answer from the formula above.
- **Lines 5 and 6** print both.

> <font color='green'>**Result**</font> The walk and the formula agree on $w\approx2.04$ MW per degree. The same loop that minimised the cost curve fits the forecast line.

### <font color='purple'>Quiz 4</font>

**Question:** Let $f(x)=(x-4)^{2}$, so $f'(x)=2(x-4)$. Gradient descent starts at $x=0$ with learning rate $\eta=0.25$. What is $x$ after one update?

**Options:**

- A) $-2$
- B) $2$
- C) $8$
- D) $4$

**Correct Answer:** B

**Explanation:**

- **A)** Wrong: $-2$ adds the slope, $0+0.25(-8)$; descent subtracts it.
- **B)** Correct: $f'(0)=-8$, so $x_{1}=0-0.25(-8)=2$.
- **C)** Wrong: $8$ is $0-(-8)$, which forgets to multiply the slope by $\eta$.
- **D)** Wrong: $4$ is the minimum itself, but one update with $\eta=0.25$ moves only part of the way.

---

## <font color='blue'>Closing: Outcome & Summary</font> `15 min`

Let us return to Meera's question and answer it using each topic once.

> <font color='purple'>**1. At 50 MW, should Meera buy more or less? (Topic 1)**</font>

The power rule gave $C'(5)=81>0$, so the cost climbs as she buys more. She should buy less. The chain rule gave $\dfrac{dC}{dT}=9.6$ lakh per degree at a 40 °C forecast.

> <font color='purple'>**2. Which way does the cost climb fastest? (Topic 2)**</font>

The gradient points that way, and its negative is the way to cut cost fastest. Let us put the results of Topics 1 to 4 in one place:

```python
print("slope at x = 5       :", dC(5.0))
print("dC/dT at T = 40      :", round(dC(x_of_T(40.0)) * dx_dT(40.0), 2))
print("steepest rise at p   :", np.round(grad_F(p) / np.linalg.norm(grad_F(p)), 3))
print("loss gradient (1, 60):", np.round(grad_loss(1.0, 60.0), 2))
x_best = descend(dC, 5.0, 0.02, 60)[-1]
print("descent ends at x    :", round(x_best, 3), "| cost", round(C(x_best), 3))
```

**Output**

```
slope at x = 5       : 81.0
dC/dT at T = 40      : 9.6
steepest rise at p   : [0.992 0.129]
loss gradient (1, 60): [-69.95 -11.15]
descent ends at x    : 2.0 | cost 22.0
```

#### <font color='#c2410c'>Code Explanation</font>

- **Line 1** is the slope at 50 MW from the power rule (Topic 1).
- **Line 2** is the chain rule at a 40 °C forecast (Topic 1).
- **Line 3** is the gradient at $p$ shrunk to length $1$: the direction of steepest ascent (Topic 2).
- **Line 4** is the squared-error gradient at the guess $w=1,\ b=60$ (Topic 3).
- **Lines 5 and 6** run 60 downhill moves from $x=5$ and print where they end (Topic 4).

> <font color='purple'>**3. Which way should the forecast line move? (Topic 3)**</font>

Both entries of the loss gradient are negative, so $w$ and $b$ should both go up. Moving against the gradient is the same walk as in Topic 4.

> <font color='purple'>**4. How much does the downhill walk save against the four tries on the sheet? (Topic 4)**</font>

```python
tries = np.array([1.0, 3.0, 5.0, 7.0])
sheet_x = tries[np.argmin(C(tries))]
gap = C(sheet_x) - C(x_best)
print("sheet  : x =", sheet_x, "| cost =", C(sheet_x))
print("descent: x =", round(x_best, 3), "| cost =", round(C(x_best), 3))
print("gap per evening:", round(gap, 3), "lakh | over 90 evenings:", round(gap * 90 / 100, 2), "crore")
```

**Output**

```
sheet  : x = 1.0 | cost = 30.0
descent: x = 2.0 | cost = 22.0
gap per evening: 8.0 lakh | over 90 evenings: 7.2 crore
```

#### <font color='#c2410c'>Code Explanation</font>

- **Line 1** is Meera's four sizes: 10, 30, 50 and 70 MW.
- **Line 2** picks the cheapest of those four (Topic 1's cost function `C`).
- **Line 3** is how much more the sheet's choice costs than the end of the downhill walk (Topic 4).
- **Lines 4 to 6** print both choices and the gap. The last line converts lakh to crore, since 100 lakh make 1 crore.

<p align="center"><img src="https://coding-platform.s3.amazonaws.com/dev/lms/tickets/135556d7-bd92-4271-80ad-c971f303d093/nHfqPtfFcSI3FEBd.png" alt="The cost curve C(x) with Meera's four sheet tries marked at x = 1, 3, 5 and 7, and the point reached by gradient descent at x = 2 with a cost of 22 lakh. The gap of 8 lakh between the best sheet try and the descent result is shown." width="900"></p>

### <font color='green'>Outcome</font>

|                         |         Four tries on a sheet         |             Gradient descent             |
| :---------------------- | :-----------------------------------: | :---------------------------------------: |
| Purchase size found     |            $x=1$ (10 MW)            |          $x\approx2.0$ (20 MW)          |
| Cost per evening        |              $30$ lakh              |            $\approx22$ lakh            |
| Extra paid per evening  |              $8$ lakh              |               $\approx0$               |
| Over 90 summer evenings |           Rs $7.2$ crore           |               $\approx0$               |
| What it needs           | Luck that a try lands near the bottom | A slope formula and a small learning rate |

> <font color='green'>**Therefore,**</font> Meera should read the slope and walk against it: at 50 MW the slope is $+81$, so buy less, and keep moving until the slope is close to $0$. Four tries missed the bottom by 8 lakh an evening. The price is that she needs a slope formula and a learning rate small enough that the walk does not jump away.

> <font color="red">**🎯 Instructor Note**</font> Reveal the opening poll: "smaller" was right, because the slope at 50 MW is positive. Ask how many learners chose it.

### <font color='green'>Summary</font>

| Learning objective                                                          | Takeaway                                                                                                                                                        |
| :-------------------------------------------------------------------------- | :-------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Compute the derivative of a polynomial function using the power rule        | $\dfrac{d}{dx}(a x^{n})=a n x^{n-1}$ term by term; a constant drops out. $C'(5)=81$.                                                                        |
| Apply the chain rule to differentiate a composite function                  | Outer slope times inner slope: $C'(x(T))\cdot x'(T)$. At 40 °C it is $48\times0.2=9.6$.                                                                    |
| Construct the gradient vector of a multivariable function at a stated point | One partial derivative per variable, others held fixed. $\nabla F(3,2)=[23,\ 3]$.                                                                             |
| Interpret the gradient as the direction of steepest ascent at a point       | The gradient points uphill, its length is the rate of rise, and the negative gradient points downhill.                                                          |
| Compute the gradient of a squared-error loss with respect to its parameters | $\partial L/\partial w=-\tfrac{2}{n}\sum x_i u_i$ and $\partial L/\partial b=-\tfrac{2}{n}\sum u_i$, with $u_i=y_i-(wx_i+b)$.                             |
| Implement gradient descent to minimise a single-variable function           | Repeat $x\leftarrow x-\eta f'(x)$ with a small $\eta$; the walk settles where the slope is $0$.                                                           |
| **Practical value**                                                   | You can find the best setting of a cost, a price or a model parameter by following the slope, and you can check any slope with a tiny step before you trust it. |

---

# <font color='red'>Launch the Feedback Poll</font>

> <font color="red">**🎯 Instructor Note**</font> Reminder: launch the feedback poll now to collect learner feedback.
