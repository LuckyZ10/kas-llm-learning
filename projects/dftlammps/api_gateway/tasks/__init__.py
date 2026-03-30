"""
任务模块
"""

from .celery_app import (
    celery_app,
    task_manager,
    update_task_status,
    send_failure_notification,
)

from .dft_tasks import (
    run_dft_calculation,
    calculate_band_structure,
    calculate_dos,
    relax_structure,
    calculate_phonon,
    run_neb_calculation,
)

from .md_tasks import (
    run_md_simulation,
    equilibrate_system,
    calculate_rdf,
    calculate_msd,
    analyze_trajectory,
    run_metadynamics,
)

from .ml_tasks import (
    train_ml_model,
    evaluate_model,
    validate_model,
    hyperparameter_search,
    active_learning_iteration,
)

from .screening_tasks import (
    run_screening,
    batch_calculate_properties,
    generate_candidates,
    optimize_composition,
)

from .analysis_tasks import (
    analyze_calculation_results,
    export_results,
    generate_report,
    compare_calculations,
    visualize_results,
    data_pipeline,
)

__all__ = [
    # Celery应用
    "celery_app",
    "task_manager",
    "update_task_status",
    "send_failure_notification",
    # DFT任务
    "run_dft_calculation",
    "calculate_band_structure",
    "calculate_dos",
    "relax_structure",
    "calculate_phonon",
    "run_neb_calculation",
    # MD任务
    "run_md_simulation",
    "equilibrate_system",
    "calculate_rdf",
    "calculate_msd",
    "analyze_trajectory",
    "run_metadynamics",
    # ML任务
    "train_ml_model",
    "evaluate_model",
    "validate_model",
    "hyperparameter_search",
    "active_learning_iteration",
    # 筛选任务
    "run_screening",
    "batch_calculate_properties",
    "generate_candidates",
    "optimize_composition",
    # 分析任务
    "analyze_calculation_results",
    "export_results",
    "generate_report",
    "compare_calculations",
    "visualize_results",
    "data_pipeline",
]
