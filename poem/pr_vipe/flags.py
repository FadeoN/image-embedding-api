class Flags:
    input_keypoint_profile_name_2d: str = 'LEGACY_2DCOCO13'
    min_input_keypoint_score_2d: float = -1.0
    embedding_type: str = 'POINT'
    embedding_size: int = 16
    num_embedding_components: int = 1
    num_embedding_samples: int = 20
    base_model_type: str = 'SIMPLE'
    num_fc_blocks: int = 2
    num_fcs_per_block: int = 2
    num_hidden_nodes: int = 1024
    num_bottleneck_nodes: int = 0
    weight_max_norm: float = 0.0
    checkpoint_path: str = "./poem/pr_vipe/checkpoint/model.ckpt-02013963"
    use_moving_average: bool = True
    master: str = ''


