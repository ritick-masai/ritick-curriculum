## <span style="color:#2F7DDE">Probability Distributions</span>

### <span style="color:#2F7DDE">Mental Map</span>

```mermaid
%%{init: {'flowchart': {'nodeSpacing': 45, 'rankSpacing': 55, 'padding': 20}}}%%
flowchart TB
  CUR["<b>Current Session</b><br/>Probability Distributions"]
  Q["<b>The Big Question</b><br/><i>How likely is a result, and when is it truly surprising?</i>"]
  CV["<b>Course use case</b><br/>Analysts decide whether a result is ordinary or surprising across 10,000 delivery records"]
  RL["<b>Real-world example</b><br/>A delivery app that counts late orders and measures waiting time"]

  CUR ==> Q
  Q ==>|&nbsp;Used in the course&nbsp;| CV
  Q ==>|&nbsp;Seen in real life&nbsp;| RL

  classDef currSes fill:#ffe0b2,stroke:#ef6c00,color:#4e342e,stroke-width:3px
  classDef ask fill:#e3f2fd,stroke:#1565c0,color:#0d47a1,stroke-width:3px
  classDef val fill:#e8f5e9,stroke:#2e7d32,color:#1b5e20

  class CUR currSes
  class Q ask
  class CV,RL val

  linkStyle default stroke-width:3px
```

---

### <span style="color:#2F7DDE">What You'll Learn</span>

Have you ever wondered how a delivery app guesses your waiting time before you order?

By the end of this session, you will be able to:

- <span style="color:#D9660B">**Classify**</span> each measurement as something you count or something you measure.
- <span style="color:#D9660B">**Calculate**</span> the chance that exactly 3 out of 10 orders arrive late.
- <span style="color:#D9660B">**Determine**</span> the chance that one delivery takes longer than 40 minutes.
- <span style="color:#D9660B">**Estimate**</span> what share of deliveries land close to the usual time.

> <span style="color:#2F7DDE">**Topics you will meet:**</span> discrete vs continuous, binomial, normal and z-scale, and the empirical rule.
>
> Join the session to learn each of these in detail.

---

### <span style="color:#2F7DDE">Why You Are Learning This Topic</span>

You order dinner, and the app promises 35 minutes. Yours arrives early. Your friend's order arrives late.

The app cannot know each wait in advance. It knows which waits are common and which are rare. A <span style="color:#D9660B">**probability distribution**</span> means a pattern that shows how likely each outcome is.

In your course, you will use the same idea to judge whether a result is ordinary or surprising.

> <span style="color:#2F7DDE">**Think about it:**</span> How does the app know that one wait is unusual?

---

### <span style="color:#2F7DDE">The Delivery App Puzzle</span>

The app promised 35 minutes. Any order over 35 minutes counts as late. Here are the last 6 orders from one restaurant.

| Order | Minutes taken |
|:-----:|:-------------:|
| 1 | 27 |
| 2 | 41 |
| 3 | 30 |
| 4 | 24 |
| 5 | 33 |
| 6 | 38 |

<span style="color:#D9660B">**What you want to happen:**</span>

- Know how likely 3 late orders out of 10 are.
- Know how unusual a 41-minute wait is.
- Know what share of orders arrive between 25 and 35 minutes.

> <span style="color:#D9660B">**Wonder about this:**</span>
>
> - Why can you count late orders, but only measure the minutes?
> - Would one 60-minute order change what you call usual?

---

### <span style="color:#2F7DDE">Think It Through</span>

There is no wrong first try. Keep your guesses for the session.

**1. Which one do you count: late orders, or minutes of waiting?**
Hint: Ask whether you could ever see 2.5 late orders.

**2. If about 30 minutes is usual, which order looks the most unusual?**
Hint: Look for the biggest gap from 30.

**3. If 10 orders go out tonight, what number of late orders would surprise you?**
Hint: Use the 2 late orders out of 6 above as a guide.

---

### <span style="color:#2F7DDE">One Question to Carry Into the Session</span>

> <span style="color:#2F7DDE">**How can a computer turn a feeling like "that wait was unusual" into an exact chance?**</span>
>
> And how can you check its answer yourself?
