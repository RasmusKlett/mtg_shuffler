# %%
import itertools
import math
import random
import statistics
import typing

import matplotlib.pyplot as plt
import numpy as np

type Sample = list[bool]


# %%
def default_distribution(num: int, land_prob=0.4) -> Sample:
    lands = [True] * round(num * land_prob)
    all_cards = lands + [False] * (num - len(lands))
    random.shuffle(all_cards)
    return all_cards


def deterministic_distribution(num: int) -> Sample:
    card_gen = itertools.cycle([True, False, True, False, False])
    return list(itertools.islice(card_gen, num))


def dota_probabilities() -> list[float]:
    """
    The distribution used by Dota 2. 40% lands on average.
    Looks a little too restrictive, maybe we want more randomness than this?

    See https://dota2.fandom.com/wiki/Random_Distribution?so=search#Pseudo_random_events
    """
    c_const = 0.201547413607754017070679639
    probabilities = []
    for k in range(1, math.ceil(1 / c_const) + 1):
        probabilities.append(
            math.factorial(k)
            * c_const
            * math.prod(1 / i - c_const for i in range(1, k))
        )
    return probabilities


def probability_to_sample_inf(prob: list[float]) -> typing.Iterable[bool]:
    random_between_lands = (
        random.choices(list(range(1, len(prob) + 1)), prob)[0]
        for _ in itertools.count()
    )
    return itertools.chain.from_iterable(
        [False] * (x - 1) + [True] for x in random_between_lands
    )


def probability_to_sample(prob: list[float], num: int) -> Sample:
    """
    Takes a probability distribution, and generates a sample of size num
    following this distribution. Enforces 40% lands, so the end of the sample
    may not follow the distribution.
    """
    # Set up infinite iterable to sample
    infinite_sample = probability_to_sample_inf(prob)

    # Draw from the iterable until we have drawn all lands and/or spells we need
    lands_missing = round(0.4 * num)
    spells_missing = num - lands_missing
    sample = []
    while lands_missing and spells_missing:
        next_card = next(infinite_sample)
        sample.append(next_card)
        if next_card:
            lands_missing -= 1
        else:
            spells_missing -= 1

    # Fill out the last of the deck with spells or lands. At least one will be 0.
    sample += [True] * lands_missing
    sample += [False] * spells_missing

    return sample


# %%


def dist_probabilities(sample: Sample):
    """
    Calculate the probability of lands and non-lands in the sample
    """
    land_empirical_prob = sum(sample) / len(sample)
    return land_empirical_prob, 1 - land_empirical_prob


def plot_int_probs(sample: Sample, ax, *args, **kwargs):
    unique_counts = np.unique(sample, return_counts=True)
    return ax.plot(
        unique_counts[0], unique_counts[1] / len(sample), marker="o", *args, **kwargs
    )


def geometric_probability_distribution(prob_succ: float, max_trials: int):
    """
    Calculate the first n geometric distribution probabilities
    """
    probabilities = []
    for k in range(1, max_trials):
        probability = ((1 - prob_succ) ** (k - 1)) * prob_succ
        probabilities.append(probability)
    return list(range(1, len(probabilities) + 1)), probabilities


# %%
def dist_stdev(sample: Sample):
    """
    Calculate the standard deviation of distance between lands in the deck.
    Also returns list of these distances.
    """
    cards_between_lands = []

    last_land_index = 0
    for index, card in enumerate(sample):
        if card:
            cards_between_lands.append(index - last_land_index)
            last_land_index = index
    return cards_between_lands, statistics.stdev(
        cards_between_lands
    )  # pstdev i stedet?


# %%
# Make samples
land_prob = 0.4
default_sample = default_distribution(num=1000, land_prob=land_prob)
default_cards_between_lands, _ = dist_stdev(default_sample)

determ_sample = deterministic_distribution(1000)

dota_probs = dota_probabilities()
dota_sample = probability_to_sample(dota_probs, 1000)

geometric_vals = geometric_probability_distribution(
    land_prob, max(default_cards_between_lands)
)
dota_geo_meaned = list(
    (a + b) / 2
    for a, b in itertools.zip_longest(geometric_vals[1], dota_probs, fillvalue=0)
)
dota_geo_sample = probability_to_sample(dota_geo_meaned, 1000)
# %%

fig, ax = plt.subplots()
plot_int_probs(default_cards_between_lands, ax, label="empirical default")
ax.plot(*geometric_vals, label="geometric")
print(dota_probs)

ax.plot(list(range(1, len(dota_probs) + 1)), dota_probs, marker="o", label="dota")
ax.plot(
    list(range(1, len(dota_geo_meaned) + 1)),
    dota_geo_meaned,
    marker="o",
    label="dota geo meaned",
)
ax.legend()
# %%

samples_to_run = {
    "DEFAULT DISTRIBUTION": default_sample,
    "DETERMINISTIC DISTRIBUTION": determ_sample,
    "DOTA DISTRIBUTION": dota_sample,
    "DOTA GEO MEANED DISTRIBUTION": dota_geo_sample,
}

for name, sample in samples_to_run.items():
    print(name)
    print("Probabilities", dist_probabilities(sample))
    cards_between_in_sample, sample_stdev = dist_stdev(sample)
    print("Mean distance", statistics.mean(cards_between_in_sample))
    print("Std. dev", sample_stdev)
    print()

# %%

# Hvor stor er sandsynligheden for 2-4 lande på en starthånd

# Hvor mange lande skal du have i decket for have mindst x
# sandsynlighed for 2-4 startlande

# Hvad er sandsynlighedsfordelingen for antal lande du har efter 5 ture


def lands_after_n_cards_distribution(dist_fun, num_simulations, num_cards):
    outcomes = []
    for _ in range(num_simulations):
        cards = dist_fun(60)[:num_cards]
        outcomes.append(sum(cards))
    print("Mean", statistics.mean(outcomes))
    xs, ys = np.unique(outcomes, return_counts=True)
    return xs, ys / ys.sum()


# %%

samples_to_run = {
    "DEFAULT DISTRIBUTION": default_distribution,
    "DOTA DISTRIBUTION": lambda x: probability_to_sample(dota_probs, x),
    "DOTA GEO MEANED DISTRIBUTION": lambda x: probability_to_sample(dota_geo_meaned, x),
}

fig, ax = plt.subplots()
cards_to_draw = 11
for name, distribution in samples_to_run.items():
    results = lands_after_n_cards_distribution(distribution, 10000, cards_to_draw)

    ax.plot(*results, label=name, marker="o")

fig.suptitle(f"Probability of drawing x lands in {cards_to_draw} cards")
ax.legend()

# %%
