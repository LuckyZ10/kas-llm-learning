"""
Complete Workflow Examples for Laboratory Automation

Demonstrates integration of all modules:
- Synthesis workflow
- Characterization workflow
- Closed-loop optimization workflow
- Full pipeline from synthesis to analysis
"""

import asyncio
import json
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional
import numpy as np

from ..lab_automation.equipment import (
    EquipmentManager, RobotArm, SynthesisEquipment, 
    CharacterizationInstrument, EquipmentType
)
from ..lab_automation.synthesis import (
    SynthesisWorkflow, PowderSynthesizer, ThinFilmDepositor
)
from ..lab_automation.characterization import (
    CharacterizationWorkflow, XRDParser, SEMParser, ElectrochemicalParser
)
from ..lab_automation.control import (
    ControlLoop, PIDController, ControlConfig, OptimizationEngine
)
from ..lab_automation.lims import (
    MockLIMSClient, SampleTracker, ExperimentLogger, DataUploader
)
from ..lab_automation.ros2_interface import (
    create_ros_interface, RobotController
)


logger = logging.getLogger(__name__)


class AutomatedSynthesisWorkflow:
    """
    Complete automated synthesis workflow
    Integrates synthesis planning, equipment control, and LIMS
    """
    
    def __init__(self):
        self.equipment_manager = EquipmentManager()
        self.synthesis_workflow = SynthesisWorkflow()
        self.lims = MockLIMSClient()
        self.sample_tracker = SampleTracker(self.lims)
        self.experiment_logger = ExperimentLogger(self.lims)
        self.data_uploader = DataUploader(self.lims)
    
    async def setup_equipment(self):
        """Initialize all equipment"""
        # Robot arm for sample handling
        robot = RobotArm(
            equipment_id="robot_001",
            name="UR5e Robot Arm",
            model="UR5e",
            manufacturer="Universal Robots",
            connection_params={'host': '192.168.1.100'}
        )
        
        # Furnace for synthesis
        furnace = SynthesisEquipment(
            equipment_id="furnace_001",
            name="High-Temperature Furnace",
            equipment_subtype="furnace",
            max_temperature=1500,
            connection_params={}
        )
        
        # Mixer
        mixer = SynthesisEquipment(
            equipment_id="mixer_001",
            name="Planetary Mixer",
            equipment_subtype="mixer",
            connection_params={}
        )
        
        # Register equipment
        self.equipment_manager.register(robot)
        self.equipment_manager.register(furnace)
        self.equipment_manager.register(mixer)
        
        # Connect all
        await self.equipment_manager.connect_all()
        
        # Initialize
        await robot.initialize()
        await furnace.initialize()
        await mixer.initialize()
        
        logger.info("Equipment setup complete")
    
    async def run_synthesis_pipeline(self,
                                    target_material: str,
                                    material_type: str = "powder") -> Dict[str, Any]:
        """
        Run complete synthesis pipeline
        
        Args:
            target_material: Target material to synthesize
            material_type: Type of material (powder, thin_film)
        
        Returns:
            Synthesis results
        """
        # Generate sample ID
        sample_id = f"{target_material}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        experiment_id = f"EXP_{sample_id}"
        
        logger.info(f"Starting synthesis pipeline for {target_material}")
        
        try:
            # Step 1: Register sample in LIMS
            await self.sample_tracker.register_sample(
                sample_id=sample_id,
                name=f"{target_material} Synthesis",
                material_type=material_type,
                created_by="automation_system"
            )
            
            # Step 2: Start experiment logging
            await self.experiment_logger.start_experiment(
                experiment_id=experiment_id,
                sample_id=sample_id,
                experiment_type="synthesis",
                operator="automation_system",
                parameters={'target_material': target_material}
            )
            
            # Step 3: Mark sample as in progress
            await self.sample_tracker.start_processing(sample_id)
            
            # Step 4: Execute synthesis
            result = await self.synthesis_workflow.synthesize_material(
                target_material=target_material,
                material_type=material_type,
                sample_id=sample_id,
                equipment_manager=self.equipment_manager
            )
            
            # Step 5: Log synthesis steps
            for step in result.get('steps', []):
                await self.experiment_logger.log_step(
                    experiment_id=experiment_id,
                    step_name=step['action'],
                    results={'status': step['status']}
                )
            
            # Step 6: Complete experiment
            await self.experiment_logger.complete_experiment(
                experiment_id=experiment_id,
                final_results=result,
                notes=f"Successfully synthesized {target_material}"
            )
            
            # Step 7: Mark sample as complete
            await self.sample_tracker.complete_processing(sample_id)
            
            # Step 8: Upload data to LIMS
            await self.data_uploader.upload_experiment_data(
                sample_id=sample_id,
                experiment_id=experiment_id,
                data=result
            )
            
            logger.info(f"Synthesis pipeline completed for {sample_id}")
            
            return {
                'success': True,
                'sample_id': sample_id,
                'experiment_id': experiment_id,
                'result': result
            }
            
        except Exception as e:
            logger.error(f"Synthesis pipeline failed: {e}")
            
            # Log failure
            await self.experiment_logger.fail_experiment(
                experiment_id=experiment_id,
                error_message=str(e)
            )
            await self.sample_tracker.fail_processing(sample_id)
            
            return {
                'success': False,
                'sample_id': sample_id,
                'error': str(e)
            }


class CharacterizationPipeline:
    """
    Automated characterization pipeline
    Runs multiple characterization techniques on a sample
    """
    
    def __init__(self):
        self.equipment_manager = EquipmentManager()
        self.char_workflow = CharacterizationWorkflow()
        self.lims = MockLIMSClient()
        self.experiment_logger = ExperimentLogger(self.lims)
    
    async def setup_equipment(self):
        """Setup characterization equipment"""
        # XRD instrument
        xrd = CharacterizationInstrument(
            equipment_id="xrd_001",
            name="X-ray Diffractometer",
            instrument_type="xrd",
            model="D8 Advance",
            manufacturer="Bruker"
        )
        
        # SEM instrument
        sem = CharacterizationInstrument(
            equipment_id="sem_001",
            name="Scanning Electron Microscope",
            instrument_type="sem",
            model="JSM-7600",
            manufacturer="JEOL"
        )
        
        # Electrochemical workstation
        echem = CharacterizationInstrument(
            equipment_id="echem_001",
            name="Potentiostat",
            instrument_type="electrochemical",
            model="CHI760E",
            manufacturer="CH Instruments"
        )
        
        # Register equipment
        self.equipment_manager.register(xrd)
        self.equipment_manager.register(sem)
        self.equipment_manager.register(echem)
        
        # Connect all
        await self.equipment_manager.connect_all()
        
        logger.info("Characterization equipment setup complete")
    
    async def characterize_sample(self,
                                 sample_id: str,
                                 techniques: List[str] = None) -> Dict[str, Any]:
        """
        Run complete characterization on a sample
        
        Args:
            sample_id: Sample ID
            techniques: List of techniques to run ['xrd', 'sem', 'cv', 'eis', 'gcd']
        
        Returns:
            Characterization results
        """
        techniques = techniques or ['xrd', 'sem']
        results = {}
        
        experiment_id = f"CHAR_{sample_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        await self.experiment_logger.start_experiment(
            experiment_id=experiment_id,
            sample_id=sample_id,
            experiment_type="characterization",
            parameters={'techniques': techniques}
        )
        
        # Get equipment instances
        xrd = self.equipment_manager.get("xrd_001")
        sem = self.equipment_manager.get("sem_001")
        echem = self.equipment_manager.get("echem_001")
        
        try:
            # Run XRD if requested
            if 'xrd' in techniques and xrd:
                logger.info(f"Running XRD for {sample_id}")
                xrd_data = await self.char_workflow.run_xrd_measurement(
                    sample_id=sample_id,
                    instrument=xrd,
                    parameters={
                        'wavelength': 1.5406,
                        'scan_range': (10, 80),
                        'scan_rate': 2.0
                    }
                )
                
                # Analyze XRD data
                parser = XRDParser()
                peaks = parser.find_peaks(xrd_data)
                phases = parser.phase_identification(peaks)
                
                results['xrd'] = {
                    'peaks': [p.to_dict() for p in peaks[:5]],
                    'phases': phases
                }
                
                await self.experiment_logger.log_step(
                    experiment_id=experiment_id,
                    step_name='xrd_measurement',
                    results=results['xrd']
                )
            
            # Run SEM if requested
            if 'sem' in techniques and sem:
                logger.info(f"Running SEM for {sample_id}")
                sem_data = await self.char_workflow.run_sem_measurement(
                    sample_id=sample_id,
                    instrument=sem,
                    parameters={
                        'magnification': 10000,
                        'voltage': 15.0
                    }
                )
                
                results['sem'] = sem_data.to_dict()
                
                await self.experiment_logger.log_step(
                    experiment_id=experiment_id,
                    step_name='sem_measurement',
                    results=results['sem']
                )
            
            # Run electrochemical if requested
            if any(t in techniques for t in ['cv', 'eis', 'gcd']) and echem:
                for tech in techniques:
                    if tech in ['cv', 'eis', 'gcd']:
                        logger.info(f"Running {tech.upper()} for {sample_id}")
                        
                        echem_data = await self.char_workflow.run_electrochemical_measurement(
                            sample_id=sample_id,
                            instrument=echem,
                            technique=tech,
                            parameters={'scan_rate': 0.1} if tech == 'cv' else {}
                        )
                        
                        parser = ElectrochemicalParser()
                        if tech == 'cv':
                            analysis = parser.analyze_cv(echem_data)
                        elif tech == 'eis':
                            analysis = parser.analyze_eis(echem_data)
                        else:
                            analysis = parser.analyze_gcd(echem_data)
                        
                        results[tech] = {
                            'data': echem_data.to_dict(),
                            'analysis': analysis
                        }
                        
                        await self.experiment_logger.log_step(
                            experiment_id=experiment_id,
                            step_name=f'{tech}_measurement',
                            results=analysis
                        )
            
            # Complete experiment
            await self.experiment_logger.complete_experiment(
                experiment_id=experiment_id,
                final_results=results
            )
            
            # Comprehensive analysis
            analysis = self.char_workflow.analyze_sample(sample_id)
            
            return {
                'success': True,
                'sample_id': sample_id,
                'experiment_id': experiment_id,
                'results': results,
                'analysis': analysis
            }
            
        except Exception as e:
            logger.error(f"Characterization failed: {e}")
            await self.experiment_logger.fail_experiment(experiment_id, str(e))
            
            return {
                'success': False,
                'sample_id': sample_id,
                'error': str(e)
            }


class ClosedLoopOptimization:
    """
    Closed-loop optimization workflow
    Combines synthesis, characterization, and feedback control
    """
    
    def __init__(self):
        self.synthesis_pipeline = AutomatedSynthesisWorkflow()
        self.char_pipeline = CharacterizationPipeline()
        self.optimization_engine = OptimizationEngine()
        self.lims = MockLIMSClient()
    
    async def setup(self):
        """Setup all components"""
        await self.synthesis_pipeline.setup_equipment()
        await self.char_pipeline.setup_equipment()
    
    async def optimize_synthesis(self,
                                target_material: str,
                                target_property: str,
                                target_value: float,
                                max_iterations: int = 10) -> Dict[str, Any]:
        """
        Run closed-loop optimization of synthesis parameters
        
        Args:
            target_material: Material to synthesize
            target_property: Property to optimize ('capacity', 'conductivity', etc.)
            target_value: Target value for the property
            max_iterations: Maximum optimization iterations
        
        Returns:
            Optimization results
        """
        logger.info(f"Starting closed-loop optimization for {target_material}")
        
        # History of experiments
        experiment_history = []
        
        # Initial parameters
        current_params = {
            'temperature': 800,
            'time': 4,
            'atmosphere': 'air'
        }
        
        for iteration in range(max_iterations):
            logger.info(f"Optimization iteration {iteration + 1}/{max_iterations}")
            
            # Step 1: Synthesize with current parameters
            synthesis_result = await self.synthesis_pipeline.run_synthesis_pipeline(
                target_material=target_material
            )
            
            if not synthesis_result['success']:
                logger.error(f"Synthesis failed: {synthesis_result.get('error')}")
                continue
            
            sample_id = synthesis_result['sample_id']
            
            # Step 2: Characterize
            char_result = await self.char_pipeline.characterize_sample(
                sample_id=sample_id,
                techniques=['xrd', 'sem', 'cv']
            )
            
            if not char_result['success']:
                logger.error(f"Characterization failed: {char_result.get('error')}")
                continue
            
            # Step 3: Evaluate property
            measured_value = self._extract_property(
                char_result['results'],
                target_property
            )
            
            # Step 4: Calculate error
            error = abs(target_value - measured_value)
            
            experiment_history.append({
                'iteration': iteration,
                'sample_id': sample_id,
                'parameters': current_params.copy(),
                'measured_value': measured_value,
                'error': error
            })
            
            logger.info(f"Iteration {iteration + 1}: {target_property} = {measured_value:.4f}, "
                       f"target = {target_value:.4f}, error = {error:.4f}")
            
            # Step 5: Check convergence
            if error < 0.05 * target_value:  # Within 5%
                logger.info(f"Converged! {target_property} = {measured_value:.4f}")
                break
            
            # Step 6: Update parameters for next iteration
            current_params = self._update_parameters(
                current_params,
                measured_value,
                target_value,
                experiment_history
            )
        
        return {
            'success': True,
            'target_material': target_material,
            'target_property': target_property,
            'target_value': target_value,
            'iterations': len(experiment_history),
            'history': experiment_history,
            'best_result': min(experiment_history, key=lambda x: x['error'])
        }
    
    def _extract_property(self,
                         results: Dict[str, Any],
                         property_name: str) -> float:
        """Extract property value from characterization results"""
        if property_name == 'capacity' and 'gcd' in results:
            return results['gcd']['analysis'].get('specific_capacity_mah_g', 0)
        elif property_name == 'conductivity' and 'eis' in results:
            return 1.0 / results['eis']['analysis'].get('charge_transfer_resistance_ohm', 1)
        elif property_name == 'crystallinity' and 'xrd' in results:
            # Use peak sharpness as proxy
            peaks = results['xrd'].get('peaks', [])
            if peaks:
                return 1.0 / peaks[0].get('fwhm', 1)
        
        return 0.0
    
    def _update_parameters(self,
                          current_params: Dict[str, Any],
                          measured_value: float,
                          target_value: float,
                          history: List[Dict]) -> Dict[str, Any]:
        """Update synthesis parameters based on results"""
        new_params = current_params.copy()
        
        # Simple gradient-based update
        if measured_value < target_value:
            # Increase temperature
            new_params['temperature'] = min(1200, current_params['temperature'] + 50)
            # Increase time
            new_params['time'] = min(24, current_params['time'] + 1)
        else:
            # Decrease temperature
            new_params['temperature'] = max(400, current_params['temperature'] - 50)
            # Decrease time
            new_params['time'] = max(1, current_params['time'] - 0.5)
        
        return new_params


class CompleteWorkflow:
    """
    Complete laboratory automation workflow
    Demonstrates full integration of all modules
    """
    
    def __init__(self):
        self.synthesis = AutomatedSynthesisWorkflow()
        self.characterization = CharacterizationPipeline()
        self.optimization = ClosedLoopOptimization()
        self.ros_interface = create_ros_interface("lab_automation_node", use_mock=True)
        self.robot_controller = RobotController(self.ros_interface)
    
    async def setup(self):
        """Setup complete workflow"""
        logger.info("Setting up complete workflow...")
        
        # Initialize ROS
        self.ros_interface.initialize()
        self.robot_controller.initialize()
        
        # Setup equipment
        await self.synthesis.setup_equipment()
        await self.characterization.setup_equipment()
        await self.optimization.setup()
        
        logger.info("Setup complete")
    
    async def run_full_pipeline(self,
                               target_material: str,
                               material_type: str = "powder",
                               run_characterization: bool = True,
                               run_optimization: bool = False) -> Dict[str, Any]:
        """
        Run complete pipeline from synthesis to analysis
        
        Args:
            target_material: Material to synthesize
            material_type: Type of material
            run_characterization: Whether to run characterization
            run_optimization: Whether to run optimization loop
        
        Returns:
            Complete results
        """
        logger.info(f"Running full pipeline for {target_material}")
        
        results = {
            'target_material': target_material,
            'material_type': material_type,
            'timestamp': datetime.now().isoformat()
        }
        
        try:
            # Step 1: Synthesis
            synthesis_result = await self.synthesis.run_synthesis_pipeline(
                target_material=target_material,
                material_type=material_type
            )
            results['synthesis'] = synthesis_result
            
            if not synthesis_result['success']:
                results['status'] = 'failed'
                results['error'] = 'Synthesis failed'
                return results
            
            sample_id = synthesis_result['sample_id']
            
            # Step 2: Robot handling (mock)
            logger.info(f"Robot handling sample {sample_id}")
            # In real implementation, robot would move sample to characterization station
            
            # Step 3: Characterization
            if run_characterization:
                char_result = await self.characterization.characterize_sample(
                    sample_id=sample_id,
                    techniques=['xrd', 'sem']
                )
                results['characterization'] = char_result
            
            # Step 4: Optimization (optional)
            if run_optimization:
                opt_result = await self.optimization.optimize_synthesis(
                    target_material=target_material,
                    target_property='capacity',
                    target_value=150.0,
                    max_iterations=3
                )
                results['optimization'] = opt_result
            
            results['status'] = 'completed'
            
        except Exception as e:
            logger.error(f"Pipeline failed: {e}")
            results['status'] = 'failed'
            results['error'] = str(e)
        
        return results
    
    def generate_report(self, results: Dict[str, Any]) -> str:
        """Generate comprehensive workflow report"""
        lines = [
            "=" * 60,
            "Laboratory Automation Workflow Report",
            "=" * 60,
            f"Target Material: {results.get('target_material')}",
            f"Material Type: {results.get('material_type')}",
            f"Timestamp: {results.get('timestamp')}",
            f"Status: {results.get('status')}",
            "",
            "Synthesis Results:",
        ]
        
        synthesis = results.get('synthesis', {})
        if synthesis.get('success'):
            lines.extend([
                f"  Sample ID: {synthesis.get('sample_id')}",
                f"  Yield: {synthesis.get('result', {}).get('yield_g', 0):.2f} g",
            ])
        else:
            lines.append(f"  Failed: {synthesis.get('error', 'Unknown error')}")
        
        char = results.get('characterization', {})
        if char and char.get('success'):
            lines.extend([
                "",
                "Characterization Results:",
            ])
            char_results = char.get('results', {})
            if 'xrd' in char_results:
                lines.append("  XRD: Completed")
            if 'sem' in char_results:
                lines.append("  SEM: Completed")
        
        if 'error' in results:
            lines.extend([
                "",
                f"Error: {results['error']}"
            ])
        
        lines.append("=" * 60)
        
        return "\n".join(lines)


# Example usage functions
async def example_synthesis():
    """Example: Run automated synthesis"""
    workflow = AutomatedSynthesisWorkflow()
    await workflow.setup_equipment()
    
    result = await workflow.run_synthesis_pipeline(
        target_material="LiCoO2",
        material_type="powder"
    )
    
    print(json.dumps(result, indent=2))
    return result


async def example_characterization():
    """Example: Run automated characterization"""
    pipeline = CharacterizationPipeline()
    await pipeline.setup_equipment()
    
    result = await pipeline.characterize_sample(
        sample_id="TEST_001",
        techniques=['xrd', 'sem']
    )
    
    print(json.dumps(result, indent=2))
    return result


async def example_full_workflow():
    """Example: Run complete workflow"""
    workflow = CompleteWorkflow()
    await workflow.setup()
    
    result = await workflow.run_full_pipeline(
        target_material="ZnO",
        material_type="thin_film",
        run_characterization=True,
        run_optimization=False
    )
    
    print(workflow.generate_report(result))
    return result


if __name__ == "__main__":
    # Run examples
    logging.basicConfig(level=logging.INFO)
    
    # Run synthesis example
    # asyncio.run(example_synthesis())
    
    # Run full workflow example
    asyncio.run(example_full_workflow())
