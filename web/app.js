"use strict";

const DOTA_C_VALUES = [
    [0.05, 0.003801658303553139101756466],
    [0.10, 0.014745844781072675877050816],
    [0.15, 0.032220914373087674975117359],
    [0.20, 0.055704042949781851858398652],
    [0.25, 0.084744091852316990275274806],
    [0.30, 0.118949192725403987583755553],
    [0.35, 0.157983098125747077557540462],
    [0.40, 0.201547413607754017070679639],
    [0.45, 0.249306998440163189714677100],
    [0.50, 0.302103025348741965169160432],
    [0.55, 0.360397850933168697104686803],
    [0.60, 0.422649730810374235490851220],
    [0.65, 0.481125478337229174401911323],
    [0.70, 0.571428571428571428571428572],
    [0.75, 0.666666666666666666666666667],
    [0.80, 0.750000000000000000000000000],
    [0.85, 0.823529411764705882352941177],
    [0.90, 0.888888888888888888888888889],
    [0.95, 0.947368421052631578947368421],
]

// Like simple Python range. Lower inclusive, upper exclusive.
const range = (start, stop) =>
    Array.from({length: stop - start}, (_, i) => start + i)

const sum = (nums) => nums.length > 0 ? nums.reduce((x, y) => x + y, 0) : 0

const product = (nums) => nums.length > 0 ? nums.reduce((x, y) => x * y, 1) : 1

const factorial = (num) => product(range(1, num + 1))

const stdev = (sample) => {
    if (sample.length == 1) {
        return 0.0
    }
    const mean = sum(sample) / sample.length
    const vari = sample.reduce((acc, value) => acc + (value - mean)**2 / (sample.length - 1), 0)
    return Math.sqrt(vari)
}

// Implements subset of Python random.choices
const randomChoice = (values, weights) => {
    if (values.length !== weights.length) {
        throw Error("random_choice values and weights must have same length")
    }
    const total_weights = sum(weights)
    const cumulative_weights = weights.slice(1).reduce(
        (cum, current) => [...cum, current / total_weights + cum[cum.length - 1]],
        [weights[0] / total_weights]
    )
    
    const rand = Math.random()
    const chosenIndex = cumulative_weights.findIndex(x => x >= rand)

    return values[chosenIndex]
}

function dotaProbabilities(c_const = 0.201547413607754017070679639) {
    console.log("wat")

    const probabilities = range(1, Math.min(Math.ceil(1 / c_const) + 1, 40)).map((k) => 
        factorial(k)
        * c_const
        * product(range(1, k).map(i => 1 / i - c_const))
    )
    console.log("probs", probabilities)
    return probabilities
}
const dotaProbabilitiesPrecomputed = DOTA_C_VALUES.map(
    entry => [entry[0], dotaProbabilities(entry[1])]
)

const nearestDotaProbabilities = meanProb => {
    let lowestDist = Math.abs(dotaProbabilitiesPrecomputed[0][0] - meanProb)
    let bestProbabilities = dotaProbabilitiesPrecomputed[0][1]

    for (const [p, probabilities] of dotaProbabilitiesPrecomputed) {
        const dist = Math.abs(p - meanProb)
        if (dist > lowestDist) {
            break
        }
        lowestDist = dist
        bestProbabilities = probabilities
    }
    return bestProbabilities
}


function adaptiveDotaDistribution(spells, lands) {
    const sample = []

    const to_draw = []
    let first = true
    
    while (spells > 0 & lands > 0) {
        if (to_draw.length == 0) {
            let probs = undefined
            if (first) {
                // console.log("first")
                first = false
                probs = [0.4, 0.314, 0.19, 0.086]
            } else {
                // console.log("Later")
                probs = nearestDotaProbabilities(lands / (lands + spells))
            }
            const distToNextLand = randomChoice(range(1, probs.length + 1), probs)
            to_draw.push(...Array(distToNextLand - 1).fill(false))
            to_draw.push(true)
        }
        const card = to_draw.shift()
        if (card) {
            lands--
        } else {
            spells--
        }
        sample.push(card)
    }
    sample.push(...Array(lands).fill(true))
    sample.push(...Array(spells).fill(false))

    return sample
}

class IntValuedInput {
    constructor(htmlElement) {
        this.htmlElement = htmlElement
    }
    get() {
        return parseInt(this.htmlElement.value)
    }
    set(value) {
        this.htmlElement.value = value
    }
}

const spellInputEl = document.getElementById("spellNum")
const landInputEl = document.getElementById("landNum")
const drawnCardsEl = document.getElementById("drawnCards")

const spellCount = new IntValuedInput(spellInputEl)
const landCount = new IntValuedInput(landInputEl)


let initialized = false
let sample = null


function drawCard() {
    if (!initialized) {
        spellInputEl.disabled = true
        landInputEl.disabled = true
        sample = adaptiveDotaDistribution(spellCount.get(), landCount.get())
        initialized = true
    }
    if (sample.length === 0) {
        return
    }

    const card = sample.shift()

    const landsLeft = sum(sample)
    landCount.set(landsLeft)
    spellCount.set(sample.length - landsLeft)
    const newElement = document.createElement("div")
    newElement.className = "card " + (card ? "land" : "spell")
    drawnCardsEl.prepend(newElement)
}

const distStdev = sample => {
    const cardsBetweenLands = []

    let last_land_index = 0
    for (const [index, card] of sample.entries()) {
        if (card) {
            cardsBetweenLands.push(index - last_land_index)
            last_land_index = index
        }
    }
    return [cardsBetweenLands, stdev(cardsBetweenLands)]
}

const meanAfterNCards = num_cards => {
    const results = Array.from({length: 10000}, _ => 
        sum(adaptiveDotaDistribution(36, 24).slice(0, num_cards))
    )
    console.log("results", results)
    return sum(results) / results.length
}

function analyse() {
    const sampleAna = adaptiveDotaDistribution(3000, 2000)
    const [cardsBetweenLands, sampleStdev] = distStdev(sampleAna)
    console.log("Mean distance", sum(cardsBetweenLands) / cardsBetweenLands.length)
    console.log("Std. deviation", sampleStdev)
    console.log("Mean lands after 11 cards", meanAfterNCards(11))

    console.log(cardsBetweenLands)
}