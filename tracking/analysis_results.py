import _init_paths
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = [8, 8]

from lib.test.analysis.plot_results import plot_results, print_results, print_per_sequence_results
from lib.test.evaluation import get_dataset, trackerlist

trackers = []
dataset_name = 'otb100_uwb'  # 使用 OTB100_UWB 数据集

"""ostrack"""
trackers.extend(trackerlist(name='ostrack', parameter_name='vitb_256_mae_32x4_ep300', dataset_name=dataset_name,
                            run_ids=None, display_name='OSTrack256'))
trackers.extend(trackerlist(name='ostrack', parameter_name='vitb_256_mae_ce_32x4_ep300', dataset_name=dataset_name,
                            run_ids=None, display_name='OSTrack256_CE'))

"""ugtrack"""
trackers.extend(trackerlist(name='ugtrack',
                            parameter_name='vitb_256_mae_ep300_uwb_conv1d_token',
                            dataset_name=dataset_name,
                            run_ids=None,
                            display_name='UGTrack_Conv1D_Token'))


dataset = get_dataset(dataset_name)

# plot_results(trackers, dataset, 'OTB2015', merge_results=True, plot_types=('success', 'norm_prec'),
#              skip_missing_seq=False, force_evaluation=True, plot_bin_gap=0.05)
print_results(trackers, dataset, dataset_name, merge_results=True, plot_types=('success', 'norm_prec', 'prec'))

