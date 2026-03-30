"""
DFT-LAMMPS Digital Twin Platform

数字孪生与实时预测系统 - Phase 50

This package provides a comprehensive digital twin platform for materials systems,
including real-time synchronization, predictive modeling, uncertainty quantification,
and visualization capabilities.
"""

__version__ = "1.0.0"
__author__ = "DFT-LAMMPS Team"

# Core modules
from .digital_twin.twin_core import (
    DigitalTwinCore,
    TwinConfiguration,
    TwinState,
    StateVector,
    Observation,
    Prediction,
    PhysicsBasedModel,
    NeuralSurrogateModel,
    ModelType,
    TwinCluster,
)

from .digital_twin.real_time_sync import (
    RealTimeSynchronizer,
    SyncConfiguration,
    SyncDirection,
    SyncMode,
    DataTransformer,
    QualityFilter,
    ExperimentDataSource,
    BidirectionalMapper,
    SyncMetrics,
)

from .digital_twin.predictive_model import (
    PredictiveMaintenanceEngine,
    DegradationModel,
    ExponentialDegradationModel,
    PowerLawDegradationModel,
    ParisLawModel,
    ParticleFilterRUL,
    HealthIndicator,
    RULPrediction,
    MaintenanceRecommendation,
    HealthLevel,
    DegradationMode,
    FailureMode,
)

from .digital_twin.uncertainty_quantification import (
    UQEngine,
    UncertaintyQuantifier,
    MonteCarloUQ,
    BootstrapUQ,
    EnsembleUQ,
    BayesianUQ,
    SensitivityAnalyzer,
    ConfidenceEstimator,
    UncertaintyPropagator,
    UncertaintyEstimate,
    ConfidenceAssessment,
    UQMethod,
    UncertaintyType,
)

# Visualization modules
from .twin_visualization.dashboard import (
    Dashboard,
    DashboardConfig,
    TimeSeriesBuffer,
    MetricValue,
    MatplotlibRenderer,
    PlotlyRenderer,
    WebDashboard,
)

from .twin_visualization.renderer_3d import (
    Renderer3D,
    MatplotlibRenderer3D,
    VolumeRenderer,
    EvolutionAnimator,
    Structure3D,
    Atom,
    Bond,
    Cell,
    FieldGrid,
    ColorMap,
)

from .twin_visualization.anomaly_alert import (
    AnomalyDetectionSystem,
    AnomalyDetector,
    StatisticalDetector,
    IsolationForestDetector,
    AutoencoderDetector,
    AlertManager,
    Alert,
    AlertLevel,
    AnomalyScore,
    AnomalyType,
    RealTimeMonitor,
)

# Example implementations
from .twin_examples.battery_twin import (
    BatteryDigitalTwin,
    BatterySpecification,
    BatteryState,
    BatteryPhysicsModel,
)

from .twin_examples.catalyst_twin import (
    CatalystDigitalTwin,
    CatalystSpecification,
    CatalystState,
    CatalystPhysicsModel,
    CatalystType,
    DeactivationMechanism,
    ActiveSite,
    ReactionCondition,
)

__all__ = [
    # Core
    "DigitalTwinCore",
    "TwinConfiguration", 
    "TwinState",
    "StateVector",
    "Observation",
    "Prediction",
    "PhysicsBasedModel",
    "NeuralSurrogateModel",
    "ModelType",
    "TwinCluster",
    
    # Sync
    "RealTimeSynchronizer",
    "SyncConfiguration",
    "SyncDirection",
    "SyncMode",
    "BidirectionalMapper",
    
    # Predictive
    "PredictiveMaintenanceEngine",
    "HealthIndicator",
    "RULPrediction",
    "HealthLevel",
    "DegradationMode",
    "FailureMode",
    
    # UQ
    "UQEngine",
    "UncertaintyEstimate",
    "UQMethod",
    
    # Visualization
    "Dashboard",
    "DashboardConfig",
    "Structure3D",
    "Atom",
    "Bond",
    "Cell",
    "AlertManager",
    "Alert",
    "AlertLevel",
    "AnomalyDetectionSystem",
    
    # Examples
    "BatteryDigitalTwin",
    "BatterySpecification",
    "CatalystDigitalTwin",
    "CatalystSpecification",
    "CatalystType",
    "DeactivationMechanism",
]


def run_all_demos():
    """运行所有演示"""
    print("=" * 80)
    print("🚀 DFT-LAMMPS 数字孪生平台 - 完整演示")
    print("=" * 80)
    
    print("\n" + "=" * 80)
    print("📦 核心模块演示")
    print("=" * 80)
    
    # Core
    from .digital_twin.twin_core import demo as core_demo
    core_demo()
    
    print("\n" + "=" * 80)
    print("🔄 实时同步演示")
    print("=" * 80)
    
    from .digital_twin.real_time_sync import demo as sync_demo
    sync_demo()
    
    print("\n" + "=" * 80)
    print("🔮 预测模型演示")
    print("=" * 80)
    
    from .digital_twin.predictive_model import demo as pred_demo
    pred_demo()
    
    print("\n" + "=" * 80)
    print("📊 不确定性量化演示")
    print("=" * 80)
    
    from .digital_twin.uncertainty_quantification import demo as uq_demo
    uq_demo()
    
    print("\n" + "=" * 80)
    print("🎨 可视化模块演示")
    print("=" * 80)
    
    from .twin_visualization.dashboard import demo as dash_demo
    dash_demo()
    
    from .twin_visualization import d_renderer as renderer_module
    renderer_module.demo()
    
    from .twin_visualization.anomaly_alert import demo as alert_demo
    alert_demo()
    
    print("\n" + "=" * 80)
    print("💡 应用案例演示")
    print("=" * 80)
    
    from .twin_examples.battery_twin import demo as battery_demo
    battery_demo()
    
    from .twin_examples.catalyst_twin import demo as catalyst_demo
    catalyst_demo()
    
    print("\n" + "=" * 80)
    print("✅ 所有演示完成!")
    print("=" * 80)


if __name__ == "__main__":
    run_all_demos()
