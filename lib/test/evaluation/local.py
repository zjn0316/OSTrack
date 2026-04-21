from lib.test.evaluation.environment import EnvSettings

def local_env_settings():
    settings = EnvSettings()

    # Set your local paths here.
    settings.prj_dir = r"D:\OSTrack"
    settings.save_dir = r"D:\OSTrack\output"
    settings.result_plot_path = r"D:\OSTrack\output\test\result_plots"
    settings.results_path = r"D:\OSTrack\output\test\tracking_results"    # Where to store tracking results
    settings.segmentation_path = r"D:\OSTrack\output\test\segmentation_results"
    settings.network_path = r"D:\OSTrack\output\test\networks"    # Where tracking networks are stored.
    settings.otb100_uwb_path = r"D:\OSTrack\data\OTB100_UWB"
    settings.davis_dir = ''
    settings.got10k_lmdb_path = ''
    settings.got10k_path = ''
    settings.got_packed_results_path = ''
    settings.got_reports_path = ''
    settings.itb_path = ''
    settings.lasot_extension_subset_path_path = ''
    settings.lasot_lmdb_path = ''
    settings.lasot_path = ''
    settings.nfs_path = ''
    settings.otb_path = ''
    settings.tc128_path = ''
    settings.tn_packed_results_path = ''
    settings.tnl2k_path = ''
    settings.tpl_path = ''
    settings.trackingnet_path = ''
    settings.uav_path = ''
    settings.vot18_path = ''
    settings.vot22_path = ''
    settings.vot_path = ''
    settings.youtubevos_dir = ''

    return settings

