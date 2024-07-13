# %%
import itertools
import math
import random
import statistics
import typing

import matplotlib.pyplot as plt
import numpy as np

type Sample = list[bool]

# Constants to scale the dota probability function to a certain mean probability.
# First value is mean probability, second is c value to achieve it.
DOTA_C_VALUES = [
    (0.05, 0.003801658303553139101756466),
    (0.10, 0.014745844781072675877050816),
    (0.15, 0.032220914373087674975117359),
    (0.20, 0.055704042949781851858398652),
    (0.25, 0.084744091852316990275274806),
    (0.30, 0.118949192725403987583755553),
    (0.35, 0.157983098125747077557540462),
    (0.40, 0.201547413607754017070679639),
    (0.45, 0.249306998440163189714677100),
    (0.50, 0.302103025348741965169160432),
    (0.55, 0.360397850933168697104686803),
    (0.60, 0.422649730810374235490851220),
    (0.65, 0.481125478337229174401911323),
    (0.70, 0.571428571428571428571428572),
    (0.75, 0.666666666666666666666666667),
    (0.80, 0.750000000000000000000000000),
    (0.85, 0.823529411764705882352941177),
    (0.90, 0.888888888888888888888888889),
    (0.95, 0.947368421052631578947368421),
]


# %%
def default_distribution(num: int, land_prob=0.4) -> Sample:
    lands = [True] * round(num * land_prob)
    all_cards = lands + [False] * (num - len(lands))
    random.shuffle(all_cards)
    return all_cards


def deterministic_distribution(num: int) -> Sample:
    card_gen = itertools.cycle([True, False, True, False, False])
    return list(itertools.islice(card_gen, num))


def dota_probabilities(c_const=0.201547413607754017070679639) -> list[float]:
    """
    The distribution used by Dota 2. 40% lands on average.
    Looks a little too restrictive, maybe we want more randomness than this?

    See https://dota2.fandom.com/wiki/Random_Distribution?so=search#Pseudo_random_events
    """

    probabilities = []
    for k in range(1, min(math.ceil(1 / c_const) + 1, 40)):
        probabilities.append(
            math.factorial(k)
            * c_const
            * math.prod(1 / i - c_const for i in range(1, k))
        )
    return probabilities


dota_probabilities_precomputed = []

for prob, c_value in DOTA_C_VALUES:
    dota_probabilities_precomputed.append((prob, dota_probabilities(c_value)))


def nearest_dota_probabilities(mean_prob: float):
    """
    Find the probabilities from dota_probabilities_precomputed that best match
    the given mean probability
    """
    lowest_dist = abs(dota_probabilities_precomputed[0][0] - mean_prob)
    best_probabilities = dota_probabilities_precomputed[0][1]

    for p, probabilities in dota_probabilities_precomputed[1:]:
        dist = abs(p - mean_prob)
        if dist > lowest_dist:
            # We found the lowest distance
            break
        lowest_dist = dist
        best_probabilities = probabilities
    return best_probabilities


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


def geometric_probability_distribution(prob_succ: float, max_trials: int):
    """
    Calculate the first n geometric distribution probabilities
    """
    probabilities = []
    for k in range(1, max_trials):
        probability = ((1 - prob_succ) ** (k - 1)) * prob_succ
        probabilities.append(probability)
    return list(range(1, len(probabilities) + 1)), probabilities


def adaptive_dota_distribution(spells: int, lands: int) -> Sample:
    """
    Make a sample using the dota distribution, but reevaluate the drawing probabilities
    after each draw, to ensure lands remain evenly spaced at end of deck even if we
    happen to draw many lands early, and vice versa.
    """
    sample = []

    to_draw = []  # Keeps track of what we have drawn from the distribution
    first = True

    # Add one card to sample per iteration
    while spells > 0 and lands > 0:
        if len(to_draw) == 0:
            # Distribution values ran out, need to take a new random with updated
            # probabilities
            if first:
                # The dota distribution is correct on average, but starts out at a very low
                # probability, which will skew the average draw noticeably for short games.
                # To counteract, we draw the first from this distribution to balance it out.
                first = False
                probs = [0.4, 0.314, 0.19, 0.086]
                # probs = [0.4, 0.32, 0.2, 0.08]
            else:
                probs = nearest_dota_probabilities(lands / (lands + spells))
            dist_to_next_land = random.choices(list(range(1, len(probs) + 1)), probs)[0]

            to_draw += [False] * (dist_to_next_land - 1)
            to_draw.append(True)

        # An attempt at reducing the clumping of lands at the end
        # if len(to_draw) > (lands + spells):
        #     final_cards = [False] * (lands + spells)
        #     for index in random.sample(range(len(final_cards)), lands):
        #         final_cards[index] = True
        #     sample += final_cards
        #     lands = 0
        #     spells = 0
        #     break

        card = to_draw.pop(0)
        if card:
            lands -= 1
        else:
            spells -= 1
        sample.append(card)

    # Fill out the last of the deck with spells or lands. At most one of these will do something
    # Most of the time these should be close to 0
    sample += [True] * lands
    sample += [False] * spells

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
default_sample = default_distribution(num=5000, land_prob=land_prob)
default_cards_between_lands, _ = dist_stdev(default_sample)

determ_sample = deterministic_distribution(5000)

dota_probs = dota_probabilities()
dota_sample = probability_to_sample(dota_probs, 5000)

adaptive_dota_sample = adaptive_dota_distribution(round(5000 * 0.6), round(5000 * 0.4))

geometric_vals = geometric_probability_distribution(
    land_prob, max(default_cards_between_lands)
)
dota_geo_meaned = list(
    (a + b) / 2
    for a, b in itertools.zip_longest(geometric_vals[1], dota_probs, fillvalue=0)
)
dota_geo_sample = probability_to_sample(dota_geo_meaned, 5000)
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
    "ADAPTIVE DOTA DISTRIBUTION": adaptive_dota_sample,
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
    "ADAPTIVE_DOTA_DISTRIBUTION": lambda x: adaptive_dota_distribution(
        round(x * 0.6), round(x * 0.4)
    ),
}

fig, ax = plt.subplots()
cards_to_draw = 11
for name, distribution in samples_to_run.items():
    results = lands_after_n_cards_distribution(distribution, 10000, cards_to_draw)

    ax.plot(*results, label=name, marker="o")

fig.suptitle(f"Probability of drawing x lands in {cards_to_draw} cards")
ax.legend()

# %%

count = 20000
# arr = np.array([probability_to_sample(dota_probs, 100) for _ in range(count)])
arr = np.array([adaptive_dota_distribution(60, 40) for _ in range(count)])
avg_per_slot = arr.sum(axis=0) / count

fig, ax = plt.subplots()
fig.suptitle(
    "Average in first half: %f" % avg_per_slot[: len(avg_per_slot) // 2].mean()
)
ax.plot(avg_per_slot, marker="x")
# ax.plot(avg_per_slot[:50], marker="x")
ax.set_xlabel("draw number")
ax.set_ylabel("Land probability")

# %%
