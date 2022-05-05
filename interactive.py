from tools import *
from matrices_modular import *
from entanglement_models import *
import numpy as np
from matplotlib import pyplot as plt
from tqdm.notebook import tqdm, trange

def random_projection(distances):
    v, s, u = torch.svd(torch.Tensor(distances))

def bin_search_inverse(f, min_bound = 1, max_bound = 100, decreasing = True, thresh = 0.01):
    def search_invert(y):
        low = min_bound
        high = max_bound

        while high - low >= thresh:
            mid = (high + low) /2
            result = f(mid)

            if (result < y) != decreasing:
                low = mid
            else:
                high = mid

        return (low + high) / 2
    return search_invert

def compute_with_p(a, b, p):
    return (a ** p + b ** p) ** (1/p)

def create_pf(a, b):
    return lambda p: compute_with_p(a, b, p)

def compute_single(matrix, i = 1, j = 1):
    f = lambda p: signed_lp_norm(torch.Tensor([
        matrix[i][0] - matrix[0][0], matrix[0][j] - matrix[0][0]
    ]), p)
    fi = bin_search_inverse(f)
    return fi(matrix[i][j] - matrix[0][0])

eps = 1e-4
def compute_single_linear(matrix, i = 1, j = 1):
    return (matrix[i][j] - matrix[0][0]) / \
        max(eps, ((matrix[i][0] - matrix[0][0]) + (matrix[0][j] - matrix[0][0])))

def score_to_distances(matrix):
    return [
        [
            compute_single(matrix, i, j)
            for j in range(1, len(matrix))
        ]
        for i in trange(1, len(matrix))
    ]

def score_to_distances_linear(matrix):
    return [
        [
            compute_single_linear(matrix, i, j)
            for j in range(1, len(matrix))
        ]
        for i in trange(1, len(matrix))
    ]


def parse(s):
    return {
        i: (i, tok, pos) for i, (tok, pos) in enumerate(
            nltk.pos_tag(nltk_tokenizer.tokenize(unidecode.unidecode(s)))
        )
    }

def markup(f):
    if len(f) == 3:
        return f[1]
    else:
        starts = ''
        ends = ''
        if 'permuted' in f[3]:
            starts += '<span style="border: 1px solid black">'
            ends  = '</span>' + ends
        if 'greenified' in f[3]:
            starts += '<span style="background-color: yellow">'
            ends += '</span>' + ends
        return starts + f[1] + ends

def markup_deparse(s):
    inverted = {
        s[i][0]: s[i]
        for i in s
    }
    flattened = [inverted[i] for i in range(len(inverted))]
    return ' '.join(markup(f) for f in flattened)

class Visualizer:
    def __init__(self, m_prop, s_prop, score_prop = None):
        self.m_prop = m_prop
        self.s_prop = s_prop
        if score_prop is None:
            self.score_prop = scored_prop_for(m_prop)
        else:
            self.score_prop = score_prop

    def visualize_matrix(self, s, show_scores = True, rows = (0, 1, 2), cols = (0, 1, 2)):
        arecs, brecs, matrix = s[self.m_prop]
        scores = s[self.score_prop]
        original = parse(matrix[0][0])

        result = '<table>'
        for i, arec in enumerate(arecs):
            if i not in rows:
                continue
            result += '<tr>'
            for j, brec in enumerate(brecs):
                if j not in cols:
                    continue
                result += '<td>'
                if show_scores:
                    result += '<b>%f: </b>' % scores[i][j]
                result += markup_deparse(
                    apply_record(
                        apply_record(
                            original, brec, markup=True
                        ),
                        arec, markup=True
                    )
                )
                result += '</td>'
            result += '</tr>'
        result += '</table>'
        return result

def extract_2d(scores, i, j, clip_outliers = 10000):
    # Filter
    scores = [x for x in scores if not any(k is None or abs(k) > clip_outliers for row in x for k in row)]

    xv = [
        x[i][0] + x[0][j] - 2 * x[0][0]
        for x in scores
    ]
    yv = [
        x[i][j] - x[0][0]
        for x in scores
    ]

    return xv, yv

eps = 1e-6

def extract_scaled_offsets(scores, i, j):
    scores = [x for x in scores if not any(k is None for row in x for k in row)]

    return [
        (x[i][j] - (x[i][0] + x[0][j] - x[0][0])) / (eps + abs(x[i][0] + x[0][j] - 2 * x[0][0]))
        for x in scores
    ]

    return xv, yv

def extract_offsets(scores, i, j):
    scores = [x for x in scores if not any(k is None for row in x for k in row)]

    return [
        x[i][j] - (x[i][0] + x[0][j] - x[0][0])
        for x in scores
    ]

    return xv, yv

def get_linear_values(scores, i, j, clip_outliers = 10000):
    xv, yv = extract_2d(scores, i, j, clip_outliers = clip_outliers)

    m = np.linalg.lstsq(
        np.expand_dims(np.array(xv), 1),
        np.array(yv),
        rcond=None
    )[0]
    mcuda = torch.Tensor(m).cuda()

    def linear_model(r, a, b):
        return r + (a + b - 2 * r) * mcuda

    return m, get_mse(linear_model, scores, i, j, clip_outliers = clip_outliers)

def plot_2d_with_line(scores, i, j, clip_outliers = 10000):
    xv, yv = extract_2d(scores, i, j, clip_outliers = clip_outliers)

    m = np.linalg.lstsq(
        np.expand_dims(np.array(xv), 1),
        np.array(yv),
        rcond=None
    )[0]
    mcuda = torch.Tensor(m).cuda()

    def linear_model(r, a, b):
        return r + (a + b - 2 * r) * mcuda

    print('Linear best fit: m=%f, mse=%f' % (
        m,
        get_mse(linear_model, scores, i, j)
    ))
    plt.scatter(xv, yv)
    plt.plot(xv, m * xv)
    plt.show()

    return linear_model

def get_properties(
        dataset_name = 'tiny-0',
        m_prop_name = 'arbitrary_3x3',
        model_label = 'scored',
        tokenizer_label = 'tok'):
    dataset = Dataset(dataset_name)
    m_prop = InjectedProperty(m_prop_name)
    scored_prop = scored_prop_for(m_prop,
            model_label = model_label,
            tokenizer_label = tokenizer_label)
    return {
        'dataset': dataset,
        'm_prop': m_prop,
        's_prop': scored_prop
    }

def extract_predicted_offsets(model, dataset, i, j):
    BATCH_SIZE = 128

    batched_dataset = [
        dataset[k:k+BATCH_SIZE]
        for k in range(len(dataset) // BATCH_SIZE + 1)
    ]

    # r, a, b, d
    batches = [
        (
            torch.Tensor([x[0][0] for x in b]),
            torch.Tensor([x[i][0] for x in b]),
            torch.Tensor([x[0][j] for x in b]),
        ) for b in batched_dataset
    ]

    criterion = torch.nn.MSELoss(reduction='sum')

    total_loss = 0
    total_examples = 0

    results = []

    with torch.no_grad():
        for r, a, b in batches:
            a, b, r = a.cuda(), b.cuda(), r.cuda()
            # Inputs a:
            results.extend(
                (
                    model(r, a, b) - (a + b - r)
                ).cpu().numpy().tolist()
            )

    return results

def get_fits(
        dataset_name = 'tiny-0',
        m_prop_name = 'arbitrary_3x3',
        model_label = 'scored',
        tokenizer_label = 'tok',
        indices = [(1, 1), (1, 2), (2, 2)],
        strings = {1: 'syntax', 2: 'semantic'},
        clip_histogram = (-250, 50),
        want_fits = True,
        pre_fn = (lambda x: x),
        want_offsets = True,
        want_offset_predictions = True,
        want_scaled_offsets = True,
        clip_outliers = 10000):

    dataset = Dataset(dataset_name)
    m_prop = InjectedProperty(m_prop_name)
    scored_prop = scored_prop_for(m_prop,
            model_label = model_label,
            tokenizer_label = tokenizer_label)
    scored = dataset[scored_prop]

    result_strings = []
    result_m = []
    result_p = []
    with torch.no_grad():
        for i, j in indices:
            pnorm_prop = entanglement_model(scored_prop, i, j,
                    PnormEntanglementModel, 'pnorm8')
            pnorm_model = dataset[pnorm_prop]

            m, m_mse = get_linear_values(scored, i, j, clip_outliers = clip_outliers)
            p, p_mse = pnorm_model.p, get_mse(pnorm_model, scored, i, j, clip_outliers = clip_outliers)

            result_strings.append('%s/%s' % (strings[i], strings[j]))
            result_m.append(m.tolist())
            result_p.append(p.cpu().numpy().tolist())

        return result_strings, result_m, result_p

def get_values(
        dataset_name = 'tiny-0',
        m_prop_name = 'arbitrary_3x3',
        model_label = 'scored',
        tokenizer_label = 'tok',
        indices = [(1, 1), (1, 2), (2, 2)],
        strings = {1: 'syntax', 2: 'semantic'},
        clip_histogram = (-250, 50),
        want_fits = True,
        pre_fn = (lambda x: x),
        want_offsets = True,
        want_offset_predictions = True,
        want_scaled_offsets = True,
        clip_outliers = 10000):

    dataset = Dataset(dataset_name)
    m_prop = InjectedProperty(m_prop_name)
    scored_prop = scored_prop_for(m_prop,
            model_label = model_label,
            tokenizer_label = tokenizer_label)
    scored = dataset[scored_prop]

    result_strings = []
    result_m = []
    result_p = []

    for i, j in indices:
        pnorm_prop = entanglement_model(scored_prop, i, j,
                PnormEntanglementModel, 'pnorm8')
        pnorm_model = dataset[pnorm_prop]

        m, m_mse = get_linear_values(scored, i, j, clip_outliers = clip_outliers)
        p, p_mse = pnorm_model.p, get_mse(pnorm_model, scored, i, j, clip_outliers = clip_outliers)

        result_strings.append('%s/%s' % (strings[i], strings[j]))
        result_m.append(m_mse.cpu().numpy().tolist())
        result_p.append(p_mse.cpu().numpy().tolist())

    return result_strings, result_m, result_p

def make_barh(kwarg_list):
    result_strings = []
    result_m = []
    result_p = []

    for name, kwargs in kwarg_list:
        strings, ms, ps = get_values(**kwargs)
        for string, m, p in zip(strings, ms, ps):
            result_strings.append(name + ': ' + string)
            result_m.append(m)
            result_p.append(p)

    width=0.35
    fig, ax = plt.subplots()
    x = np.arange(len(result_strings))

    ax.barh(x - width / 2, result_p, width, label='Best Signed Lp Norm')
    ax.barh(x + width / 2, result_m, width, label='Best Linear')
    ax.set_yticks(x)
    ax.set_yticklabels(result_strings)
    ax.set_xlabel('Mean squared error')
    ax.legend()
    plt.show()

def generate_plots(
        dataset_name = 'tiny-0',
        m_prop_name = 'arbitrary_3x3',
        model_label = 'scored',
        tokenizer_label = 'tok',
        indices = [(1, 1), (1, 2), (2, 2)],
        strings = {1: 'syntax', 2: 'semantic'},
        clip_histogram = (-250, 50),
        want_fits = True,
        pre_fn = (lambda x: x),
        want_offsets = True,
        want_offset_predictions = True,
        want_scaled_offsets = True,
        clip_outliers = 10000):

    dataset = Dataset(dataset_name)
    m_prop = InjectedProperty(m_prop_name)
    scored_prop = scored_prop_for(m_prop,
            model_label = model_label,
            tokenizer_label = tokenizer_label)
    scored = dataset[scored_prop]

    decided_p = {}
    decided_m = {}

    for i, j in indices:
        pnorm_prop = entanglement_model(scored_prop, i, j,
                PnormEntanglementModel, 'pnorm8')
        pnorm_model = dataset[pnorm_prop]
        decided_p[i, j] = pnorm_model

        if want_fits or True:
            print('PLOT FOR: %s with %s' % (strings[i], strings[j]))
            print('Pnorm best fit: p=%f, mse=%f' % (
                pnorm_model.p,
                get_mse(pnorm_model, scored, i, j)
            ))
            decided_m[i, j] = plot_2d_with_line(scored, i, j, clip_outliers = clip_outliers)

    if want_offsets:
        print('THREE HISTOGRAMS')
        for i, j in indices:
            plt.hist(extract_offsets(scored, i, j), bins=128, range=clip_histogram)
            plt.show()

        for i, j in indices:
            print('For interaction %s/%s' % (strings[i], strings[j]))
            offsets = [pre_fn(x) for x in extract_offsets(scored, i, j)]

            steps = 128
            high = clip_histogram[1]
            low = clip_histogram[0]
            binned = {
                p: sum(1 for x in offsets if p * (high - low) / steps + low < x < (p + 1) * (high - low) / steps + low)
                for p in range(steps)
            }

            bin_values = np.array([p * (high - low) / steps + low for p in range(steps)])
            bin_contents = np.array([binned[p] for p in range(steps)])

            # Window smoothing
            bin_contents = sum(np.roll(bin_contents, n) for n in range(-2, 3)) / 5
            plt.plot(bin_values, bin_contents, label='%s/%s' % (strings[i], strings[j]))

            mode = max(binned, key = lambda x: bin_contents[x]) * (high - low) / steps + low
            print('Mode:', mode)
            mean = sum(offsets) / len(offsets)
            print('Percent above mode:', sum(1 for x in offsets if x > mode) / len(offsets))
            print('Mean:', mean)
            print('Variance:', sum((x - mean) ** 2 for x in offsets) / len(offsets))

        plt.legend()
        plt.show()

    if want_scaled_offsets:
        for i, j in indices:
            print('For interaction %s/%s' % (strings[i], strings[j]))
            offsets = [pre_fn(x) for x in extract_scaled_offsets(scored, i, j)]

            steps = 128
            high = clip_histogram[1]
            low = clip_histogram[0]
            binned = {
                p: sum(1 for x in offsets if p * (high - low) / steps + low < x < (p + 1) * (high - low) / steps + low)
                for p in range(steps)
            }

            bin_values = np.array([p * (high - low) / steps + low for p in range(steps)])
            bin_contents = np.array([binned[p] for p in range(steps)])

            # Window smoothing
            bin_contents = sum(np.roll(bin_contents, n) for n in range(-2, 3)) / 5
            plt.plot(bin_values, bin_contents, label='%s/%s' % (strings[i], strings[j]))

            mode = max(binned, key = lambda x: bin_contents[x]) * (high - low) / steps + low
            print('Mode:', mode)
            mean = sum(offsets) / len(offsets)
            print('Percent above mode:', sum(1 for x in offsets if x > mode) / len(offsets))
            print('Mean:', mean)
            print('Variance:', sum((x - mean) ** 2 for x in offsets) / len(offsets))

        plt.legend()
        plt.show()

    if want_offset_predictions:
        print('PNORM')
        for i, j in indices:
            print('For interaction %s/%s' % (strings[i], strings[j]))
            offsets = extract_predicted_offsets(decided_p[i, j], scored, i, j)

            steps = 128
            high = clip_histogram[1]
            low = clip_histogram[0]
            binned = {
                p: sum(1 for x in offsets if p * (high - low) / steps + low < x < (p + 1) * (high - low) / steps + low)
                for p in range(steps)
            }

            bin_values = np.array([p * (high - low) / steps + low for p in range(steps)])
            bin_contents = np.array([binned[p] for p in range(steps)])

            # Window smoothing
            bin_contents = sum(np.roll(bin_contents, n) for n in range(-2, 3)) / 5
            plt.plot(bin_values, bin_contents, label='%s/%s' % (strings[i], strings[j]))

            mode = max(binned, key = lambda x: bin_contents[x]) * (high - low) / steps + low
            print('Mode:', mode)
            mean = sum(offsets) / len(offsets)
            print('Percent above mode:', sum(1 for x in offsets if x > mode) / len(offsets))
            print('Mean:', mean)
            print('Variance:', sum((x - mean) ** 2 for x in offsets) / len(offsets))

        plt.legend()
        plt.show()

        print('LINEAR')

        for i, j in indices:
            print('For interaction %s/%s' % (strings[i], strings[j]))
            offsets = extract_predicted_offsets(decided_m[i, j], scored, i, j)

            steps = 128
            high = clip_histogram[1]
            low = clip_histogram[0]
            binned = {
                p: sum(1 for x in offsets if p * (high - low) / steps + low < x < (p + 1) * (high - low) / steps + low)
                for p in range(steps)
            }

            bin_values = np.array([p * (high - low) / steps + low for p in range(steps)])
            bin_contents = np.array([binned[p] for p in range(steps)])

            # Window smoothing
            bin_contents = sum(np.roll(bin_contents, n) for n in range(-2, 3)) / 5
            plt.plot(bin_values, bin_contents, label='%s/%s' % (strings[i], strings[j]))

            mode = max(binned, key = lambda x: bin_contents[x]) * (high - low) / steps + low
            print('Mode:', mode)
            mean = sum(offsets) / len(offsets)
            print('Percent above mode:', sum(1 for x in offsets if x > mode) / len(offsets))
            print('Mean:', mean)
            print('Variance:', sum((x - mean) ** 2 for x in offsets) / len(offsets))

        plt.legend()
        plt.show()

    return scored_prop
