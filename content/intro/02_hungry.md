+++
title = "Chibany is hungry"
weight = 2
+++


![chibany laying down](images/chibanylayingdown.png) 

Chibany wakes up from dreaming of the delicious meals he will get later today. Twice per day, a student brings a bento box with a meal as an offering to Chibany. One student brings him a bento box in the early afternoon for lunch and a different student brings him a bento box in the evening for dinner. The meal is either a Hamburger {{% icon burger white %}}
 or a Tonkatsu (pork cutlet) {{% icon piggy-bank white %}}. To keep track of his meal possibilities, his lists out the four possibilities:

```mermaid
block-beta
    block
        columns 2
        a["H(amburger) H(amburger)"] b["H(amburger) T(onkastu)"]
        c["T(onkastu) H(amburger)"] d["T(onkastu) T(onkastu)"]
    end
```

## Sets

This forms a [set](./06_glossary.md/#set) of four elements. A set is a collection of elements or members. In this case, an element is defined by the two meals given to Chibany that day. Sets are defined by the elements they do or do not contain. The elements are listed with commas between them and "$\\{$" denotes the start of a set and "$\\}$" the end of a set. 

## Outcome Space

In the context of probability theory, the basic elements of what can occur are called *outcomes*. Outcomes are the fundamental building blocks probabilities are from. As they are fundamental, the Greek letter $\Omega$ is frequently used to refer to this set of possibile *outcomes*. Diligently noting his daily offerings, Chibany defines $\Omega = \\{HH, HT, TH, TT \\} $. The first letter defines his lunch offering, and the second letter defines his dinner offering. He notes that $H$ now always refers to hamburgers and $T$ to tonkatsu.

Note that technically, the elements of a set are unique. So, if Chibany writes down getting a pair of hamburgers twice and a hamburger and a tonkatsu ($\\{HH, HH, HT\\}$), he's gotten the same set of possibilities as if he only got one pair of hamburgers and a hamburger and tonkatsu ($\\{HH, HT\\}$). In other words, $\\{HH, HH, HT\\} = \\{HH, HT\\}$. 

Chibany is skeptical, but will try to keep it in mind. It can be confusing!

## Possibilities vs. Events
So far, we have discussed sets, possibile outcomes and the set of all possible outcomes $\Omega$. Chibany is interested in the set of possible meals that include Tonkatsu. What is this set?

$\{HT, TH, TT\}$

This is an example of an [event](./06_glossary.md/#event). Technically, an event or a set that is none, some, or all of the possible outcomes. 

### Quick Check

Is $\Omega$ an event? 

{{% expand "solution" %}} Yes -- it is the event that contains all possible outcomes.  {{% /expand %}}

Is $\Omega$ the set of all possible events?

{{% expand "solution" %}} No {{% /expand %}}

What is the set of all possible events for Chibany's situation?

{{%expand "solution" %}}
$\\{ \\{ \\}, \\{ HH \\}, \\{ HT\\}, \\{TH \\}, \\{TT\\}, 
\\{HH,HT\\}, \\{HH,TH\\}, \\{HH,TT\\},
\\{HT, TH\\}, \\{HT, TT \\},
\\{TH, TT\\}
\\{HH, HT, TH\\}, \\{HH, HT, TT \\}, \\{HH, TH, TT\\},
\\{HT, TH, TT\\},
\\{HH, HT, TH, TT\\}  \\}$

Note that $\\{ \\}$ is called the empty or null set and is a special set that contains no elements.
{{% /expand %}}

|[prev: goals](./01_goals.md) | [next: counting](./03_prob_count.md)|
| :--- | ---: |