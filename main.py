# %%
import math
import numpy as np
import random
import statistics
import itertools
import matplotlib.pyplot as plt

# %%
def default_distribution(num, land_prob=0.4):
    lands = [True] * round(num * land_prob)
    all_cards = lands + [False] * (num - len(lands))
    random.shuffle(all_cards)
    return all_cards
    # return random.choices([True, False], weights=[land_prob, 1-land_prob], k=num)

def deterministic_distribution(num):
    card_gen = itertools.cycle([True, False, True, False, False])
    return list(itertools.islice(card_gen, num))

def dota_probabilities():
    """
    The distribution used by Dota 2. 40% lands on average.
    Looks a little too restrictive, maybe we want more randomness than this?

    See https://dota2.fandom.com/wiki/Random_Distribution?so=search#Pseudo_random_events
    """
    c_const = 0.201547413607754017070679639 
    probabilities = []
    for k in range(1, math.ceil(1/c_const) + 1):
        probabilities.append(math.factorial(k) * c_const * math.prod(1/i - c_const for i in range(1, k)))
    return probabilities

def probability_to_sample(prob, num):
    random_between_lands = (random.choices(list(range(1, len(prob)+1)), prob)[0] for _ in itertools.count())
    chained = itertools.chain.from_iterable([False] * (x-1) + [True] for x in random_between_lands)
    return list(itertools.islice(chained, num))

# %%

def dist_probabilities(sample):
    """
    Calculate the probability of lands and non-lands in the sample
    """
    land_empirical_prob = sum(sample) / len(sample)
    return land_empirical_prob, 1 - land_empirical_prob

def plot_int_probs(sample, ax, *args, **kwargs):
    unique_counts = np.unique(sample, return_counts=True)
    return ax.plot(unique_counts[0], unique_counts[1] / len(sample), marker='o', *args, **kwargs)


def geometric_probability_distribution(prob_succ, max_trials):
    """
    Calculate the first n geometric distribution probabilities
    """
    probabilities = []
    for k in range(1, max_trials):
        probability = ((1 - prob_succ)**(k-1)) * prob_succ
        probabilities.append(probability)
    return list(range(1, len(probabilities)+1)), probabilities
# %%
def dist_stdev(sample):
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
    return cards_between_lands, statistics.stdev(cards_between_lands) # pstdev i stedet?
# %%
print("DEFAULT DISTRIBUTION")
land_prob = 0.4
default_sample = default_distribution(num=1000, land_prob=land_prob)
print("Probabilities", dist_probabilities(default_sample))
cards_between_lands, stdev = dist_stdev(default_sample)
print("Mean distance", statistics.mean(cards_between_lands))
print("Std. dev", stdev)

# %%
print("DETERMINISTIC DISTRIBUTION")
determ_sample = deterministic_distribution(1000)
print("Probabilities", dist_probabilities(determ_sample))
cards_between_determ, determ_stdev = dist_stdev(determ_sample)
print("Mean distance", statistics.mean(cards_between_determ))
print("Std. dev", determ_stdev)

# %%
print("DOTA DISTRIBUTION")
dota_probs = dota_probabilities()
dota_sample = probability_to_sample(dota_probs, 1000)
print("Probabilities", dist_probabilities(dota_sample))
cards_between_determ, determ_stdev = dist_stdev(determ_sample)
print("Mean distance", statistics.mean(cards_between_determ))
print("Std. dev", determ_stdev)
# %%
print("DOTA GEO MEANED DISTRIBUTION")
geometric_vals = geometric_probability_distribution(land_prob, max(cards_between_lands))
dota_geo_meaned = list((a + b)/2 for a, b in itertools.zip_longest(geometric_vals[1], dota_probs, fillvalue=0))
dota_geo_sample = probability_to_sample(dota_geo_meaned, 1000)
print("Probabilities", dist_probabilities(determ_sample))
cards_between_determ, determ_stdev = dist_stdev(determ_sample)
print("Mean distance", statistics.mean(cards_between_determ))
print("Std. dev", determ_stdev)
# %%

fig, ax = plt.subplots()
plot_int_probs(cards_between_lands, ax, label="empirical default")
ax.plot(*geometric_vals, label="geometric")
print(dota_probs)

ax.plot(list(range(1, len(dota_probs) + 1)), dota_probs, marker='o', label="dota")
ax.plot(list(range(1, len(dota_geo_meaned) + 1)), dota_geo_meaned, marker='o', label="dota geo meaned")
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
