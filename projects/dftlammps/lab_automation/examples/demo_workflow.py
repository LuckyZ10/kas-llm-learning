"""
Example usage scripts for the Lab Automation module

This file provides practical examples demonstrating:
1. Basic equipment control
2. Synthesis planning
3. Characterization data parsing
4. Feedback control
5. LIMS integration
6. ROS2 robot control
7. Complete workflows
"""

import asyncio
import logging
import numpy as np
from datetime import datetime

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# =============================================================================
# Example 1: Basic Equipment Control
# =============================================================================

async def example_equipment_control():
    """Example: Control laboratory equipment"""
    from dftlammps.lab_automation.equipment import (
        EquipmentManager, RobotArm, SynthesisEquipment, CharacterizationInstrument
    )
    
    logger.info("=" * 60)
    logger.info("Example 1: Equipment Control")
    logger.info("=" * 60)
    
    # Create equipment manager
    manager = EquipmentManager()
    
    # Create robot arm
    robot = RobotArm(
        equipment_id="robot_001",
        name="UR5e Robot Arm",
        model="UR5e",
        manufacturer="Universal Robots",
        dof=6,
        connection_params={'host': '192.168.1.100', 'port': 30002}
    )
    
    # Create furnace
    furnace = SynthesisEquipment(
        equipment_id="furnace_001",
        name="High-Temp Furnace",
        equipment_subtype="furnace",
        max_temperature=1500,
        connection_params={}
    )
    
    # Create XRD instrument
    xrd = CharacterizationInstrument(
        equipment_id="xrd_001",
        name="X-ray Diffractometer",
        instrument_type="xrd"
    )
    
    # Register equipment
    manager.register(robot)
    manager.register(furnace)
    manager.register(xrd)
    
    logger.info(f"Registered equipment: {manager.list_all()}")
    
    # Connect and initialize
    await manager.connect_all()
    await robot.initialize()
    await furnace.initialize()
    await xrd.initialize()
    
    # Move robot
    from dftlammps.lab_automation.equipment import Position3D
    target_pos = Position3D(x=0.3, y=0.2, z=0.5)
    await robot.move_to(target_pos)
    
    # Control furnace
    await furnace.set_temperature(800, ramp_rate=10)
    temp = await furnace.get_temperature()
    logger.info(f"Current temperature: {temp:.1f}°C")
    
    # Run XRD measurement
    xrd_data = await xrd.run_measurement(
        parameters={
            'sample_id': 'TEST_001',
            'wavelength': 1.5406,
            'scan_range': (10, 80)
        },
        duration=30
    )
    logger.info(f"XRD measurement complete: {len(xrd_data['two_theta'])} points")
    
    # Health check
    health = await manager.health_check_all()
    logger.info(f"Equipment health: {health}")
    
    logger.info("Equipment control example complete!")
    return manager


# =============================================================================
# Example 2: Synthesis Planning
# =============================================================================

async def example_synthesis_planning():
    """Example: Synthesis planning and execution"""
    from dftlammps.lab_automation.synthesis import (
        PowderSynthesizer, ThinFilmDepositor, ChemicalComponent
    )
    
    logger.info("=" * 60)
    logger.info("Example 2: Synthesis Planning")
    logger.info("=" * 60)
    
    # Powder synthesis example
    powder_synth = PowderSynthesizer()
    
    logger.info("Available powder recipes:")
    for recipe_id in powder_synth.list_recipes():
        recipe = powder_synth.get_recipe(recipe_id)
        logger.info(f"  - {recipe.name} ({recipe_id})")
    
    # Execute synthesis
    result = await powder_synth.execute_recipe(
        recipe_id="LCO_SSD_001",
        sample_id="LCO_001",
        equipment_manager=None
    )
    
    logger.info(f"Synthesis result: {result}")
    
    # Calculate yield
    yield_info = powder_synth.calculate_yield(
        recipe_id="LCO_SSD_001",
        actual_yield_g=0.85
    )
    logger.info(f"Yield analysis: {yield_info}")
    
    # Thin film deposition example
    film_depositor = ThinFilmDepositor()
    
    logger.info("\nAvailable thin film recipes:")
    for recipe_id in film_depositor.list_recipes():
        recipe = film_depositor.get_recipe(recipe_id)
        logger.info(f"  - {recipe.name} ({recipe_id})")
    
    # Optimize parameters
    optimized_params = film_depositor.optimize_parameters(
        target_properties={'thickness_nm': 100, 'uniformity_percent': 95},
        constraints={'max_temp': 400}
    )
    logger.info(f"Optimized parameters: {optimized_params}")
    
    logger.info("Synthesis planning example complete!")
    return powder_synth


# =============================================================================
# Example 3: Characterization Data Parsing
# =============================================================================

def example_characterization_parsing():
    """Example: Parse and analyze characterization data"""
    from dftlammps.lab_automation.characterization import (
        XRDParser, SEMParser, ElectrochemicalParser,
        XRDData, CVData, DataAggregator
    )
    
    logger.info("=" * 60)
    logger.info("Example 3: Characterization Data Parsing")
    logger.info("=" * 60)
    
    # Create mock XRD data
    two_theta = np.linspace(10, 80, 1401)
    intensity = np.random.normal(100, 20, len(two_theta))
    # Add peaks
    for peak_pos in [25, 38, 44, 65]:
        intensity += 500 * np.exp(-((two_theta - peak_pos) / 2)**2)
    
    xrd_data = XRDData(
        two_theta=two_theta,
        intensity=intensity,
        sample_id="TEST_XRD_001"
    )
    
    # Analyze XRD data
    xrd_parser = XRDParser()
    peaks = xrd_parser.find_peaks(xrd_data, min_intensity=150)
    
    logger.info(f"Found {len(peaks)} peaks:")
    for peak in peaks[:5]:
        logger.info(f"  2θ = {peak.position:.2f}°, "
                   f"I = {peak.intensity:.1f}, "
                   f"FWHM = {peak.fwhm:.3f}°, "
                   f"d = {peak.d_spacing:.4f} Å")
        
        # Calculate crystallite size
        size = xrd_parser.calculate_crystallite_size(peak)
        logger.info(f"    Crystallite size: {size:.1f} nm")
    
    # Phase identification
    phases = xrd_parser.phase_identification(peaks)
    logger.info(f"Identified phases: {phases}")
    
    # Electrochemical data analysis
    ec_parser = ElectrochemicalParser()
    
    # Mock CV data
    potential = np.linspace(-0.5, 1.0, 500)
    current = np.sin(potential * 5) * 1e-4 + np.random.normal(0, 1e-5, 500)
    
    cv_data = CVData(
        potential_v=potential,
        current_a=current,
        sample_id="TEST_CV_001",
        scan_rate_vs=0.1
    )
    
    cv_analysis = ec_parser.analyze_cv(cv_data)
    logger.info(f"\nCV Analysis:")
    logger.info(f"  Anodic peak: {cv_analysis['anodic_peak_potential_v']:.3f} V, "
               f"{cv_analysis['anodic_peak_current_a']:.2e} A")
    logger.info(f"  Cathodic peak: {cv_analysis['cathodic_peak_potential_v']:.3f} V, "
               f"{cv_analysis['cathodic_peak_current_a']:.2e} A")
    logger.info(f"  Peak separation: {cv_analysis['peak_separation_v']:.3f} V")
    logger.info(f"  Reversible: {cv_analysis['reversible']}")
    
    # Data aggregation
    aggregator = DataAggregator()
    aggregator.add_data("TEST_SAMPLE_001", "xrd", xrd_data)
    aggregator.add_data("TEST_SAMPLE_001", "cv", cv_data)
    
    correlations = aggregator.correlate_properties("TEST_SAMPLE_001")
    logger.info(f"\nCorrelated properties: {correlations}")
    
    logger.info("Characterization parsing example complete!")
    return aggregator


# =============================================================================
# Example 4: Feedback Control
# =============================================================================

async def example_feedback_control():
    """Example: PID control for temperature regulation"""
    from dftlammps.lab_automation.control import (
        PIDController, ControlConfig, ControlLoop
    )
    
    logger.info("=" * 60)
    logger.info("Example 4: Feedback Control")
    logger.info("=" * 60)
    
    # Create PID controller
    config = ControlConfig(
        name="Temperature Controller",
        sample_time_s=1.0,
        output_limits=(0, 100)  # 0-100% power
    )
    
    pid = PIDController(
        config=config,
        kp=2.0,
        ki=0.5,
        kd=0.1
    )
    
    # Simulation parameters
    setpoint = 800.0  # Target temperature
    process_value = 25.0  # Current temperature
    
    # Simulate process (first-order system)
    def simulate_process(current_temp, control_output, dt):
        """Simulate temperature response"""
        gain = 10.0  # °C/% power
        time_constant = 60.0  # seconds
        
        target = control_output * gain + 25  # Ambient offset
        dtemp = (target - current_temp) / time_constant
        return current_temp + dtemp * dt
    
    logger.info(f"Control simulation: Target = {setpoint}°C")
    logger.info(f"{'Time (s)':<10} {'PV (°C)':<12} {'Output (%)':<12} {'Error':<10}")
    
    # Run simulation
    for t in range(120):  # 2 minutes
        # Update controller
        signal = pid.update(setpoint, process_value, dt=1.0)
        
        # Log every 10 seconds
        if t % 10 == 0:
            logger.info(f"{t:<10} {process_value:<12.2f} "
                       f"{signal.value:<12.1f} {setpoint - process_value:<10.2f}")
        
        # Simulate process response
        process_value = simulate_process(process_value, signal.value, 1.0)
    
    # Get performance metrics
    metrics = pid.get_performance_metrics()
    logger.info(f"\nControl Performance:")
    logger.info(f"  IAE: {metrics['IAE']:.2f}")
    logger.info(f"  MAE: {metrics['MAE']:.2f}°C")
    logger.info(f"  Max error: {metrics['max_error']:.2f}°C")
    
    logger.info("Feedback control example complete!")
    return pid


# =============================================================================
# Example 5: LIMS Integration
# =============================================================================

async def example_lims_integration():
    """Example: LIMS integration for sample tracking"""
    from dftlammps.lab_automation.lims import (
        MockLIMSClient, SampleTracker, ExperimentLogger, DataUploader
    )
    
    logger.info("=" * 60)
    logger.info("Example 5: LIMS Integration")
    logger.info("=" * 60)
    
    # Create LIMS client
    lims = MockLIMSClient(base_url="mock://lims.local")
    await lims.connect()
    
    # Create sample tracker
    tracker = SampleTracker(lims)
    
    # Register samples
    sample_id = f"SAMPLE_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    await tracker.register_sample(
        sample_id=sample_id,
        name="LiCoO2 Synthesis Batch 1",
        material_type="cathode_material",
        created_by="Researcher_A",
        project="Battery_Research_2024",
        priority="high"
    )
    
    logger.info(f"Registered sample: {sample_id}")
    
    # Track sample lifecycle
    await tracker.start_processing(sample_id)
    logger.info(f"Sample status: {tracker.get_sample_status(sample_id).name}")
    
    # Log experiment
    logger_exp = ExperimentLogger(lims)
    
    experiment_id = f"EXP_{sample_id}"
    await logger_exp.start_experiment(
        experiment_id=experiment_id,
        sample_id=sample_id,
        experiment_type="solid_state_synthesis",
        operator="automation_system",
        parameters={
            'temperature': 900,
            'time': 12,
            'atmosphere': 'air'
        }
    )
    
    # Log steps
    await logger_exp.log_step(
        experiment_id=experiment_id,
        step_name="weighing",
        parameters={'Li2CO3': 739, 'Co3O4': 2408}
    )
    
    await logger_exp.log_step(
        experiment_id=experiment_id,
        step_name="calcination",
        parameters={'temperature': 900, 'duration': 12}
    )
    
    # Complete experiment
    await logger_exp.complete_experiment(
        experiment_id=experiment_id,
        final_results={'yield_g': 2.8, 'purity': 0.98},
        notes="Synthesis completed successfully"
    )
    
    # Upload data
    uploader = DataUploader(lims)
    await uploader.upload_experiment_data(
        sample_id=sample_id,
        experiment_id=experiment_id,
        data={
            'xrd_pattern': {'peaks': [25.4, 38.2, 44.5]},
            'particle_size': {'d50': 5.2, 'd90': 12.8}
        }
    )
    
    # Complete sample processing
    await tracker.complete_processing(sample_id)
    logger.info(f"Final sample status: {tracker.get_sample_status(sample_id).name}")
    
    # Generate report
    report = logger_exp.generate_report(experiment_id)
    logger.info(f"\nExperiment Report:\n{report}")
    
    logger.info("LIMS integration example complete!")
    return lims


# =============================================================================
# Example 6: ROS2 Robot Control
# =============================================================================

async def example_ros2_control():
    """Example: ROS2 robot control"""
    from dftlammps.lab_automation.ros2_interface import (
        create_ros_interface, RobotController, RobotPose
    )
    
    logger.info("=" * 60)
    logger.info("Example 6: ROS2 Robot Control")
    logger.info("=" * 60)
    
    # Create ROS2 interface (mock mode for demonstration)
    ros = create_ros_interface("lab_robot_node", use_mock=True)
    ros.initialize()
    
    # Create robot controller
    robot = RobotController(ros, robot_name="ur5e")
    robot.initialize()
    
    # Define target poses
    poses = [
        RobotPose.from_euler(0.3, 0.2, 0.5, 0, 0, 0),
        RobotPose.from_euler(0.4, 0.1, 0.4, 0, 0, np.pi/4),
        RobotPose.from_euler(0.2, 0.3, 0.6, 0, np.pi/6, 0),
    ]
    
    logger.info("Moving robot through trajectory...")
    for i, pose in enumerate(poses):
        logger.info(f"  Moving to pose {i+1}: {pose.to_dict()}")
        success = await robot.move_to_pose(pose)
        if success:
            logger.info(f"  Pose {i+1} reached")
        else:
            logger.warning(f"  Failed to reach pose {i+1}")
    
    # Joint control
    logger.info("\nMoving joints...")
    joint_positions = [0.5, -0.5, 0.3, -0.3, 0.2, -0.2]
    await robot.move_joints(joint_positions, duration=2.0)
    
    logger.info("ROS2 control example complete!")
    return robot


# =============================================================================
# Example 7: Complete Workflow
# =============================================================================

async def example_complete_workflow():
    """Example: Complete automated workflow"""
    from dftlammps.lab_automation.workflows import CompleteWorkflow
    
    logger.info("=" * 60)
    logger.info("Example 7: Complete Workflow")
    logger.info("=" * 60)
    
    # Create workflow
    workflow = CompleteWorkflow()
    
    # Setup
    await workflow.setup()
    
    # Run full pipeline
    result = await workflow.run_full_pipeline(
        target_material="TiO2",
        material_type="powder",
        run_characterization=True,
        run_optimization=False
    )
    
    # Generate report
    report = workflow.generate_report(result)
    logger.info(f"\n{report}")
    
    logger.info("Complete workflow example finished!")
    return result


# =============================================================================
# Main execution
# =============================================================================

async def run_all_examples():
    """Run all examples"""
    
    examples = [
        ("Equipment Control", example_equipment_control),
        ("Synthesis Planning", example_synthesis_planning),
        ("Characterization Parsing", lambda: example_characterization_parsing()),
        ("Feedback Control", example_feedback_control),
        ("LIMS Integration", example_lims_integration),
        ("ROS2 Control", example_ros2_control),
        ("Complete Workflow", example_complete_workflow),
    ]
    
    results = {}
    
    for name, example_func in examples:
        logger.info("\n" + "=" * 60)
        logger.info(f"Running: {name}")
        logger.info("=" * 60)
        
        try:
            if asyncio.iscoroutinefunction(example_func):
                result = await example_func()
            else:
                result = example_func()
            results[name] = {"success": True, "result": result}
            logger.info(f"✓ {name} completed successfully")
        except Exception as e:
            results[name] = {"success": False, "error": str(e)}
            logger.error(f"✗ {name} failed: {e}")
    
    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("Example Execution Summary")
    logger.info("=" * 60)
    
    for name, result in results.items():
        status = "✓ PASS" if result["success"] else "✗ FAIL"
        logger.info(f"{status}: {name}")
    
    total = len(results)
    passed = sum(1 for r in results.values() if r["success"])
    logger.info(f"\nTotal: {passed}/{total} examples passed")
    
    return results


if __name__ == "__main__":
    # Run all examples
    asyncio.run(run_all_examples())
