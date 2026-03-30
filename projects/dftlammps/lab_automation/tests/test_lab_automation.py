"""
Unit tests for Lab Automation module
"""

import pytest
import asyncio
import numpy as np
from datetime import datetime

from dftlammps.lab_automation.equipment import (
    EquipmentManager, RobotArm, SynthesisEquipment, 
    CharacterizationInstrument, EquipmentType, Position3D
)
from dftlammps.lab_automation.synthesis import (
    PowderSynthesizer, ThinFilmDepositor, SynthesisRecipe,
    ChemicalComponent, PowderSynthesisParameters
)
from dftlammps.lab_automation.characterization import (
    XRDParser, SEMParser, ElectrochemicalParser,
    XRDData, CVData, EISData, DataAggregator
)
from dftlammps.lab_automation.control import (
    PIDController, ControlConfig, ControlLoop, OptimizationEngine
)
from dftlammps.lab_automation.lims import (
    MockLIMSClient, SampleTracker, SampleStatus, SampleMetadata
)


# =============================================================================
# Equipment Tests
# =============================================================================

@pytest.mark.asyncio
async def test_robot_arm():
    """Test robot arm functionality"""
    robot = RobotArm(
        equipment_id="test_robot",
        name="Test Robot",
        model="UR5e"
    )
    
    # Test connection
    assert await robot.connect()
    assert robot._connected
    
    # Test initialization
    assert await robot.initialize()
    assert robot.status.state.name == "IDLE"
    
    # Test movement
    target = Position3D(0.3, 0.2, 0.5)
    assert await robot.move_to(target)
    
    current_pos = await robot.get_position()
    assert current_pos is not None
    assert current_pos.x == target.x
    
    # Test emergency stop
    assert await robot.emergency_stop()
    
    # Test disconnection
    assert await robot.disconnect()
    assert not robot._connected


@pytest.mark.asyncio
async def test_synthesis_equipment():
    """Test synthesis equipment functionality"""
    furnace = SynthesisEquipment(
        equipment_id="test_furnace",
        name="Test Furnace",
        equipment_subtype="furnace",
        max_temperature=1000
    )
    
    assert await furnace.connect()
    assert await furnace.initialize()
    
    # Test temperature control
    await furnace.set_temperature(500, ramp_rate=50)
    temp = await furnace.get_temperature()
    assert abs(temp - 500) < 1.0
    
    # Test process
    assert await furnace.start_process(duration=1, parameters={'test': True})
    assert not furnace._process_running
    
    await furnace.disconnect()


@pytest.mark.asyncio
async def test_equipment_manager():
    """Test equipment manager"""
    manager = EquipmentManager()
    
    robot = RobotArm(equipment_id="robot1", name="Robot 1")
    furnace = SynthesisEquipment(equipment_id="furnace1", name="Furnace 1", equipment_subtype="furnace")
    
    # Register equipment
    assert manager.register(robot)
    assert manager.register(furnace)
    assert len(manager.list_all()) == 2
    
    # Get equipment
    assert manager.get("robot1") == robot
    assert manager.get("furnace1") == furnace
    
    # Get by type
    robots = manager.get_by_type(EquipmentType.ROBOT_ARM)
    assert len(robots) == 1
    assert robots[0] == robot
    
    # Connect all
    results = await manager.connect_all()
    assert all(results.values())
    
    # Health check
    health = await manager.health_check_all()
    assert len(health) == 2
    
    # Unregister
    assert manager.unregister("robot1")
    assert len(manager.list_all()) == 1


# =============================================================================
# Synthesis Tests
# =============================================================================

@pytest.mark.asyncio
async def test_powder_synthesizer():
    """Test powder synthesizer"""
    synthesizer = PowderSynthesizer()
    
    # Check default recipes
    recipes = synthesizer.list_recipes()
    assert len(recipes) > 0
    
    # Get recipe
    recipe = synthesizer.get_recipe(recipes[0])
    assert recipe is not None
    assert isinstance(recipe, SynthesisRecipe)
    
    # Execute recipe
    result = await synthesizer.execute_recipe(
        recipe_id=recipes[0],
        sample_id="TEST_001",
        equipment_manager=None
    )
    
    assert result['status'] == 'completed'
    assert 'steps' in result
    
    # Calculate yield
    yield_info = synthesizer.calculate_yield(recipes[0], 0.9)
    assert 'yield_percent' in yield_info


def test_chemical_component():
    """Test chemical component calculations"""
    component = ChemicalComponent(
        name="Test Compound",
        formula="ABC",
        purity=0.99,
        mass_mg=1000,
        molar_mass=100
    )
    
    assert component.moles == 0.01  # 1g / 100g/mol = 0.01 mol


def test_synthesis_recipe():
    """Test synthesis recipe serialization"""
    recipe = SynthesisRecipe(
        recipe_id="TEST_001",
        name="Test Recipe",
        target_material="TestMaterial",
        method="solid_state"
    )
    
    # Convert to dict
    data = recipe.to_dict()
    assert data['recipe_id'] == "TEST_001"
    
    # Calculate stoichiometry
    stoich = recipe.calculate_stoichiometry()
    assert isinstance(stoich, dict)


# =============================================================================
# Characterization Tests
# =============================================================================

def test_xrd_parser():
    """Test XRD data parsing"""
    parser = XRDParser()
    
    # Create mock data
    two_theta = np.linspace(10, 80, 701)
    intensity = np.ones(701) * 100
    intensity[100] = 500  # Add a peak
    intensity[200] = 600
    
    xrd_data = XRDData(
        two_theta=two_theta,
        intensity=intensity,
        sample_id="TEST_XRD"
    )
    
    # Validate
    assert parser.validate(xrd_data)
    
    # Find peaks
    peaks = parser.find_peaks(xrd_data, min_intensity=200)
    assert len(peaks) >= 2
    
    # Calculate crystallite size
    size = parser.calculate_crystallite_size(peaks[0])
    assert size > 0
    
    # Phase identification
    phases = parser.phase_identification(peaks)
    assert isinstance(phases, list)


def test_electrochemical_parser():
    """Test electrochemical data parsing"""
    parser = ElectrochemicalParser()
    
    # Mock CV data
    potential = np.linspace(-0.5, 1.0, 100)
    current = np.sin(potential * 5) * 1e-4
    
    cv_data = CVData(
        potential_v=potential,
        current_a=current,
        sample_id="TEST_CV"
    )
    
    # Validate
    assert parser.validate(cv_data)
    
    # Analyze
    analysis = parser.analyze_cv(cv_data)
    assert 'anodic_peak_potential_v' in analysis
    assert 'cathodic_peak_potential_v' in analysis
    assert 'peak_separation_v' in analysis


def test_data_aggregator():
    """Test data aggregation"""
    aggregator = DataAggregator()
    
    # Add XRD data
    xrd_data = XRDData(
        two_theta=np.linspace(10, 80, 100),
        intensity=np.random.rand(100),
        sample_id="SAMPLE_001"
    )
    aggregator.add_data("SAMPLE_001", "xrd", xrd_data)
    
    # Add CV data
    cv_data = CVData(
        potential_v=np.linspace(0, 1, 50),
        current_a=np.random.rand(50) * 1e-4,
        sample_id="SAMPLE_001"
    )
    aggregator.add_data("SAMPLE_001", "cv", cv_data)
    
    # Get sample data
    sample_data = aggregator.get_sample_data("SAMPLE_001")
    assert "xrd" in sample_data
    assert "cv" in sample_data
    
    # Correlate properties
    correlations = aggregator.correlate_properties("SAMPLE_001")
    assert "techniques" in correlations


# =============================================================================
# Control Tests
# =============================================================================

def test_pid_controller():
    """Test PID controller"""
    config = ControlConfig(
        name="Test PID",
        sample_time_s=0.1,
        output_limits=(0, 100)
    )
    
    pid = PIDController(config, kp=2.0, ki=0.5, kd=0.1)
    
    # Test update
    signal = pid.update(setpoint=100, process_value=50, dt=0.1)
    assert signal.value > 0
    assert signal.value <= 100
    
    # Test multiple updates
    for _ in range(10):
        signal = pid.update(setpoint=100, process_value=60, dt=0.1)
    
    # Check history
    history = pid.get_history()
    assert len(history) > 0
    
    # Get metrics
    metrics = pid.get_performance_metrics()
    assert 'MAE' in metrics
    
    # Reset
    pid.reset()
    assert len(pid.get_history()) == 0


def test_optimization_engine():
    """Test optimization engine"""
    engine = OptimizationEngine()
    
    # Define objective function (quadratic)
    def objective(x):
        return (x[0] - 5)**2 + (x[1] - 3)**2
    
    # Optimize
    result = engine.optimize(
        objective_function=objective,
        initial_params=np.array([0.0, 0.0]),
        bounds=[(0, 10), (0, 10)],
        max_iter=50
    )
    
    assert result['success']
    assert abs(result['optimal_params'][0] - 5) < 0.1
    assert abs(result['optimal_params'][1] - 3) < 0.1


# =============================================================================
# LIMS Tests
# =============================================================================

@pytest.mark.asyncio
async def test_lims_client():
    """Test LIMS client"""
    lims = MockLIMSClient()
    
    assert await lims.connect()
    assert lims._connected
    
    # Create sample
    metadata = SampleMetadata(
        sample_id="TEST_001",
        name="Test Sample",
        material_type="test",
        created_by="test_user"
    )
    
    sample_id = await lims.create_sample(metadata)
    assert sample_id == "TEST_001"
    
    # Get sample
    retrieved = await lims.get_sample(sample_id)
    assert retrieved is not None
    assert retrieved.name == "Test Sample"
    
    # Update status
    assert await lims.update_sample_status(sample_id, SampleStatus.IN_PROGRESS)
    
    # Query experiments
    experiments = await lims.query_experiments({'sample_id': sample_id})
    assert isinstance(experiments, list)
    
    await lims.disconnect()


@pytest.mark.asyncio
async def test_sample_tracker():
    """Test sample tracker"""
    lims = MockLIMSClient()
    await lims.connect()
    
    tracker = SampleTracker(lims)
    
    # Register sample
    sample_id = "TRACKER_TEST_001"
    await tracker.register_sample(
        sample_id=sample_id,
        name="Tracker Test",
        material_type="test",
        created_by="test"
    )
    
    # Check status
    status = tracker.get_sample_status(sample_id)
    assert status == SampleStatus.REGISTERED
    
    # Start processing
    await tracker.start_processing(sample_id)
    assert tracker.get_sample_status(sample_id) == SampleStatus.IN_PROGRESS
    
    # Complete
    await tracker.complete_processing(sample_id)
    assert tracker.get_sample_status(sample_id) == SampleStatus.COMPLETED
    
    # List active
    active = tracker.list_active_samples()
    assert sample_id not in active  # Completed samples not active


# =============================================================================
# Integration Tests
# =============================================================================

@pytest.mark.asyncio
async def test_integration_synthesis_and_characterization():
    """Integration test: synthesis followed by characterization"""
    
    # Synthesis
    synthesizer = PowderSynthesizer()
    
    result = await synthesizer.execute_recipe(
        recipe_id="LCO_SSD_001",
        sample_id="INT_TEST_001",
        equipment_manager=None
    )
    
    assert result['status'] == 'completed'
    
    # Characterization (mock)
    xrd_data = XRDData(
        two_theta=np.linspace(10, 80, 100),
        intensity=np.random.rand(100) * 100,
        sample_id="INT_TEST_001"
    )
    
    parser = XRDParser()
    peaks = parser.find_peaks(xrd_data)
    
    assert len(peaks) >= 0


@pytest.mark.asyncio
async def test_integration_control_and_equipment():
    """Integration test: control loop with equipment"""
    
    # Create furnace
    furnace = SynthesisEquipment(
        equipment_id="ctrl_furnace",
        name="Control Test Furnace",
        equipment_subtype="furnace"
    )
    
    await furnace.connect()
    await furnace.initialize()
    
    # Create PID controller
    config = ControlConfig(
        name="Temperature Control",
        sample_time_s=1.0,
        output_limits=(0, 100)
    )
    pid = PIDController(config, kp=5.0, ki=1.0, kd=0.5)
    
    # Simulate control
    setpoint = 500
    current_temp = 25
    
    for _ in range(10):
        signal = pid.update(setpoint, current_temp)
        # Simulate heating
        current_temp += signal.value * 0.5
    
    assert current_temp > 25  # Temperature should increase
    
    metrics = pid.get_performance_metrics()
    assert metrics['MAE'] > 0


# =============================================================================
# Performance Tests
# =============================================================================

@pytest.mark.benchmark
@pytest.mark.asyncio
async def test_large_scale_synthesis(benchmark):
    """Benchmark: Large scale synthesis planning"""
    synthesizer = PowderSynthesizer()
    
    async def run_synthesis():
        results = []
        for i in range(10):
            result = await synthesizer.execute_recipe(
                recipe_id="LCO_SSD_001",
                sample_id=f"BENCH_{i:03d}",
                equipment_manager=None
            )
            results.append(result)
        return results
    
    results = await run_synthesis()
    assert len(results) == 10


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
