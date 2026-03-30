
class ElectrochemicalParser(DataParser):
    """Parser for electrochemical measurement data"""
    
    def __init__(self):
        super().__init__("Electrochemical Parser")
        self.supported_formats = ['.txt', '.csv', '.mpt', '.mpr']
    
    def parse(self, filepath: Union[str, Path]) -> Union[CVData, EISData, GCDData]:
        """Parse electrochemical data"""
        filepath = Path(filepath)
        technique = self._detect_technique(filepath)
        
        logger.info(f"Parsing {technique} file: {filepath}")
        
        try:
            if technique == 'CV':
                return self._parse_cv(filepath)
            elif technique == 'EIS':
                return self._parse_eis(filepath)
            elif technique == 'GCD':
                return self._parse_gcd(filepath)
            else:
                raise ValueError(f"Unknown technique: {technique}")
        except Exception as e:
            logger.error(f"Failed to parse electrochemical file: {e}")
            raise
    
    def _detect_technique(self, filepath: Path) -> str:
        """Detect measurement technique from file content"""
        with open(filepath, 'r', errors='ignore') as f:
            content = f.read().lower()
        
        if 'impedance' in content or 'eis' in content or 'freq' in content:
            return 'EIS'
        elif 'potential' in content and 'current' in content and 'scan' in content:
            return 'CV'
        elif 'charge' in content or 'discharge' in content:
            return 'GCD'
        return 'CV'
    
    def _parse_cv(self, filepath: Path) -> CVData:
        """Parse cyclic voltammetry data"""
        data = np.loadtxt(filepath, delimiter=',', skiprows=1)
        return CVData(
            potential_v=data[:, 0],
            current_a=data[:, 1],
            sample_id=filepath.stem
        )
    
    def _parse_eis(self, filepath: Path) -> EISData:
        """Parse EIS data"""
        data = np.loadtxt(filepath, delimiter=',', skiprows=1)
        return EISData(
            frequency_hz=data[:, 0],
            z_real_ohm=data[:, 1],
            z_imag_ohm=data[:, 2],
            sample_id=filepath.stem
        )
    
    def _parse_gcd(self, filepath: Path) -> GCDData:
        """Parse galvanostatic charge/discharge data"""
        data = np.loadtxt(filepath, delimiter=',', skiprows=1)
        return GCDData(
            time_s=data[:, 0],
            voltage_v=data[:, 1],
            sample_id=filepath.stem
        )
    
    def validate(self, data: ElectrochemicalData) -> bool:
        """Validate electrochemical data"""
        return isinstance(data, ElectrochemicalData) and data.sample_id != ""
    
    def analyze_cv(self, data: CVData) -> Dict[str, Any]:
        """Analyze CV data"""
        max_current_idx = np.argmax(data.current_a)
        min_current_idx = np.argmin(data.current_a)
        
        anodic_peak_v = data.potential_v[max_current_idx]
        anodic_peak_a = data.current_a[max_current_idx]
        cathodic_peak_v = data.potential_v[min_current_idx]
        cathodic_peak_a = data.current_a[min_current_idx]
        
        peak_separation_v = abs(anodic_peak_v - cathodic_peak_v)
        peak_ratio = abs(anodic_peak_a / cathodic_peak_a)
        
        return {
            'anodic_peak_potential_v': anodic_peak_v,
            'anodic_peak_current_a': anodic_peak_a,
            'cathodic_peak_potential_v': cathodic_peak_v,
            'cathodic_peak_current_a': cathodic_peak_a,
            'peak_separation_v': peak_separation_v,
            'peak_current_ratio': peak_ratio,
            'reversible': peak_separation_v < 0.059 and 0.9 < peak_ratio < 1.1
        }
    
    def analyze_eis(self, data: EISData) -> Dict[str, Any]:
        """Analyze EIS data"""
        r_solution = data.z_real_ohm[np.argmax(data.frequency_hz)]
        r_total = data.z_real_ohm[np.argmin(data.frequency_hz)]
        r_ct = r_total - r_solution
        
        min_imag_idx = np.argmin(data.z_imag_ohm)
        f_max = data.frequency_hz[min_imag_idx]
        z_imag_max = abs(data.z_imag_ohm[min_imag_idx])
        c = 1 / (2 * np.pi * f_max * z_imag_max)
        
        return {
            'solution_resistance_ohm': r_solution,
            'charge_transfer_resistance_ohm': r_ct,
            'capacitance_f': c,
            'characteristic_frequency_hz': f_max
        }
    
    def analyze_gcd(self, data: GCDData) -> Dict[str, Any]:
        """Analyze GCD data for battery/supercapacitor metrics"""
        discharge_time = data.time_s[-1] - data.time_s[0]
        discharge_capacity_ah = data.current_a * discharge_time / 3600
        specific_capacity_mah_g = (discharge_capacity_ah * 1000) / data.mass_active_material_g
        avg_voltage = np.mean(data.voltage_v)
        energy_density_wh_kg = (specific_capacity_mah_g * avg_voltage) / 1000
        discharge_power_w = data.current_a * avg_voltage
        power_density_w_kg = discharge_power_w / (data.mass_active_material_g / 1000)
        
        return {
            'discharge_time_s': discharge_time,
            'discharge_capacity_mah': discharge_capacity_ah * 1000,
            'specific_capacity_mah_g': specific_capacity_mah_g,
            'energy_density_wh_kg': energy_density_wh_kg,
            'power_density_w_kg': power_density_w_kg,
            'average_voltage_v': avg_voltage
        }


class DataAggregator:
    """Aggregates and correlates data from multiple characterization techniques"""
    
    def __init__(self):
        self.parsers = {
            'xrd': XRDParser(),
            'sem': SEMParser(),
            'electrochemical': ElectrochemicalParser()
        }
        self.data_store: Dict[str, Dict[str, Any]] = {}
    
    def add_data(self, sample_id: str, technique: str, data: Any) -> bool:
        """Add characterization data for a sample"""
        if sample_id not in self.data_store:
            self.data_store[sample_id] = {}
        
        self.data_store[sample_id][technique] = {
            'data': data,
            'timestamp': datetime.now().isoformat()
        }
        return True
    
    def get_sample_data(self, sample_id: str) -> Dict[str, Any]:
        """Get all data for a sample"""
        return self.data_store.get(sample_id, {})
    
    def correlate_properties(self, sample_id: str) -> Dict[str, Any]:
        """Correlate properties from different techniques"""
        sample_data = self.get_sample_data(sample_id)
        
        correlations = {
            'sample_id': sample_id,
            'techniques': list(sample_data.keys()),
            'properties': {}
        }
        
        if 'sem' in sample_data:
            sem_analysis = sample_data['sem']['data']
            if isinstance(sem_analysis, ParticleAnalysis):
                correlations['properties']['particle_size_nm'] = sem_analysis.mean_size_nm
        
        if 'xrd' in sample_data:
            xrd_data = sample_data['xrd']['data']
            if isinstance(xrd_data, XRDData):
                parser = self.parsers['xrd']
                peaks = parser.find_peaks(xrd_data)
                if peaks:
                    crystallite_size = parser.calculate_crystallite_size(peaks[0])
                    correlations['properties']['crystallite_size_nm'] = crystallite_size
        
        if 'cv' in sample_data or 'gcd' in sample_data or 'eis' in sample_data:
            ec_data = sample_data.get('cv') or sample_data.get('gcd') or sample_data.get('eis')
            if ec_data:
                parser = self.parsers['electrochemical']
                data = ec_data['data']
                if isinstance(data, CVData):
                    correlations['properties']['cv_analysis'] = parser.analyze_cv(data)
                elif isinstance(data, GCDData):
                    correlations['properties']['gcd_analysis'] = parser.analyze_gcd(data)
                elif isinstance(data, EISData):
                    correlations['properties']['eis_analysis'] = parser.analyze_eis(data)
        
        return correlations
    
    def generate_report(self, sample_id: str) -> str:
        """Generate comprehensive characterization report"""
        correlations = self.correlate_properties(sample_id)
        
        lines = [
            f"Characterization Report for Sample: {sample_id}",
            "=" * 60,
            f"Techniques Used: {', '.join(correlations['techniques'])}",
            "",
            "Properties:",
        ]
        
        for prop, value in correlations['properties'].items():
            if isinstance(value, dict):
                lines.append(f"  {prop}:")
                for k, v in value.items():
                    lines.append(f"    {k}: {v:.4e}" if isinstance(v, float) else f"    {k}: {v}")
            else:
                lines.append(f"  {prop}: {value:.2f}" if isinstance(value, float) else f"  {prop}: {value}")
        
        return "\n".join(lines)
    
    def export_to_json(self, sample_id: str, filepath: str) -> bool:
        """Export all data to JSON"""
        try:
            data = self.get_sample_data(sample_id)
            serializable_data = {}
            for technique, info in data.items():
                if hasattr(info['data'], 'to_dict'):
                    serializable_data[technique] = info['data'].to_dict()
                else:
                    serializable_data[technique] = info
            
            with open(filepath, 'w') as f:
                json.dump(serializable_data, f, indent=2)
            return True
        except Exception as e:
            logger.error(f"Failed to export data: {e}")
            return False


class CharacterizationWorkflow:
    """High-level workflow for characterization tasks"""
    
    def __init__(self):
        self.aggregator = DataAggregator()
        self.active_measurements: Dict[str, Dict[str, Any]] = {}
    
    async def run_xrd_measurement(self,
                                  sample_id: str,
                                  instrument: Any,
                                  parameters: Dict[str, Any]) -> XRDData:
        """Run XRD measurement workflow"""
        logger.info(f"Starting XRD measurement for {sample_id}")
        
        # Load sample
        await instrument.load_sample(sample_id)
        
        # Run measurement
        data = await instrument.run_measurement(parameters)
        
        # Unload sample
        await instrument.unload_sample()
        
        # Parse and store
        xrd_data = XRDData(
            two_theta=np.array(data['two_theta']),
            intensity=np.array(data['intensity']),
            sample_id=sample_id,
            wavelength=parameters.get('wavelength', 1.5406),
            scan_rate=parameters.get('scan_rate', 2.0)
        )
        
        self.aggregator.add_data(sample_id, 'xrd', xrd_data)
        return xrd_data
    
    async def run_sem_measurement(self,
                                  sample_id: str,
                                  instrument: Any,
                                  parameters: Dict[str, Any]) -> SEMData:
        """Run SEM measurement workflow"""
        logger.info(f"Starting SEM measurement for {sample_id}")
        
        await instrument.load_sample(sample_id)
        data = await instrument.run_measurement(parameters)
        await instrument.unload_sample()
        
        sem_data = SEMData(
            sample_id=sample_id,
            magnification=parameters.get('magnification', 10000),
            accelerating_voltage_kv=parameters.get('voltage', 15.0),
            working_distance_mm=parameters.get('working_distance', 10.0)
        )
        
        self.aggregator.add_data(sample_id, 'sem', sem_data)
        return sem_data
    
    async def run_electrochemical_measurement(self,
                                              sample_id: str,
                                              instrument: Any,
                                              technique: str,
                                              parameters: Dict[str, Any]) -> Union[CVData, EISData, GCDData]:
        """Run electrochemical measurement workflow"""
        logger.info(f"Starting {technique} measurement for {sample_id}")
        
        await instrument.load_sample(sample_id)
        data = await instrument.run_measurement(parameters)
        await instrument.unload_sample()
        
        if technique == 'cv':
            ec_data = CVData(
                potential_v=np.array(data['potential']),
                current_a=np.array(data['current']),
                sample_id=sample_id,
                scan_rate_vs=parameters.get('scan_rate', 0.1)
            )
            self.aggregator.add_data(sample_id, 'cv', ec_data)
        elif technique == 'eis':
            ec_data = EISData(
                frequency_hz=np.array(data['frequency']),
                z_real_ohm=np.array(data['z_real']),
                z_imag_ohm=np.array(data['z_imag']),
                sample_id=sample_id
            )
            self.aggregator.add_data(sample_id, 'eis', ec_data)
        else:
            ec_data = GCDData(
                time_s=np.array(data['time']),
                voltage_v=np.array(data['voltage']),
                sample_id=sample_id,
                current_a=parameters.get('current', 0.001)
            )
            self.aggregator.add_data(sample_id, 'gcd', ec_data)
        
        return ec_data
    
    def analyze_sample(self, sample_id: str) -> Dict[str, Any]:
        """Comprehensive analysis of all data for a sample"""
        return self.aggregator.correlate_properties(sample_id)
    
    def generate_full_report(self, sample_id: str) -> str:
        """Generate full characterization report"""
        return self.aggregator.generate_report(sample_id)
