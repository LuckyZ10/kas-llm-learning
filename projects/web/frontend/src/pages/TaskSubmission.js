import React, { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import {
  Box,
  Stepper,
  Step,
  StepLabel,
  Button,
  Typography,
  Paper,
  TextField,
  Grid,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  FormControlLabel,
  Switch,
  Slider,
  Divider,
  Alert,
  Chip,
  Autocomplete,
  CircularProgress,
} from '@mui/material';
import {
  Science as ScienceIcon,
  Calculate as CalculateIcon,
  Memory as MemoryIcon,
  Assessment as AssessmentIcon,
} from '@mui/icons-material';
import { useTaskStore, usePresetStore, useStructureStore } from '../store/authStore';
import api from '../utils/api';

const steps = ['Structure', 'DFT Settings', 'ML Training', 'MD Simulation'];

const TaskSubmission = () => {
  const navigate = useNavigate();
  const { submitTask, isLoading, error } = useTaskStore();
  const { presets, fetchPresets } = usePresetStore();
  const { structures, searchStructures } = useStructureStore();
  
  const [activeStep, setActiveStep] = useState(0);
  const [searchQuery, setSearchQuery] = useState('');
  const [submitError, setSubmitError] = useState(null);
  
  const [formData, setFormData] = useState({
    name: '',
    description: '',
    workflow_type: 'full_workflow',
    
    // Structure
    material_id: '',
    formula: '',
    
    // DFT Config
    dft_config: {
      code: 'vasp',
      functional: 'PBE',
      encut: 520,
      kpoints_density: 0.25,
      ncores: 4,
    },
    
    // ML Config
    ml_config: {
      framework: 'deepmd',
      preset: 'fast',
      num_models: 4,
    },
    
    // MD Config
    md_config: {
      ensemble: 'nvt',
      temperatures: [300, 500, 700],
      timestep: 1.0,
      nsteps_prod: 100000,
    },
    
    // Analysis Config
    analysis_config: {
      compute_diffusion: true,
      compute_conductivity: true,
      compute_activation_energy: true,
    },
    
    // Stage control
    skip_dft: false,
    skip_ml: false,
    skip_md: false,
    skip_analysis: false,
    
    priority: 5,
  });
  
  useEffect(() => {
    fetchPresets();
  }, [fetchPresets]);
  
  const handleNext = () => {
    setActiveStep((prev) => prev + 1);
  };
  
  const handleBack = () => {
    setActiveStep((prev) => prev - 1);
  };
  
  const handleChange = (section, field, value) => {
    setFormData((prev) => ({
      ...prev,
      [section]: section.includes('_config')
        ? { ...prev[section], [field]: value }
        : value,
    }));
  };
  
  const handleStructureSearch = async (query) => {
    if (query.length >= 2) {
      await searchStructures(query);
    }
  };
  
  const handleSubmit = async () => {
    setSubmitError(null);
    const result = await submitTask(formData);
    
    if (result) {
      navigate(`/tasks/${result.task_id}`);
    } else {
      setSubmitError(error || 'Failed to submit task');
    }
  };
  
  const applyPreset = (presetName) => {
    const preset = presets?.default_configs?.[presetName];
    if (!preset) return;
    
    setFormData((prev) => ({
      ...prev,
      dft_config: { ...prev.dft_config, ...preset.dft },
      ml_config: { ...prev.ml_config, ...preset.ml },
      md_config: { ...prev.md_config, ...preset.md },
    }));
  };
  
  const renderStructureStep = () => (
    <Grid container spacing={3}>
      <Grid item xs={12}>
        <Typography variant="h6" gutterBottom>
          Structure Selection
        </Typography>
        <Typography variant="body2" color="textSecondary" gutterBottom>
          Search for a material from Materials Project or enter a formula/material ID.
        </Typography>
      </Grid>
      
      <Grid item xs={12} md={6}>
        <Autocomplete
          freeSolo
          options={structures}
          getOptionLabel={(option) => 
            typeof option === 'string' ? option : `${option.formula} (${option.material_id})`
          }
          onInputChange={(e, value) => {
            setSearchQuery(value);
            handleStructureSearch(value);
          }}
          onChange={(e, value) => {
            if (value && typeof value === 'object') {
              setFormData((prev) => ({
                ...prev,
                material_id: value.material_id,
                formula: value.formula,
              }));
            }
          }}
          renderInput={(params) => (
            <TextField
              {...params}
              label="Search Materials (formula or mp-XXXX)"
              placeholder="e.g., Li3PS4 or mp-1234"
              helperText="Enter at least 2 characters to search"
            />
          )}
        />
      </Grid>
      
      <Grid item xs={12} md={6}>
        <TextField
          fullWidth
          label="Or enter Material ID directly"
          value={formData.material_id}
          onChange={(e) => handleChange('', 'material_id', e.target.value)}
          placeholder="e.g., mp-1234"
        />
      </Grid>
      
      <Grid item xs={12}>
        <Divider sx={{ my: 2 }} />
        <Typography variant="subtitle1" gutterBottom>Task Information</Typography>
      </Grid>
      
      <Grid item xs={12} md={6}>
        <TextField
          fullWidth
          required
          label="Task Name"
          value={formData.name}
          onChange={(e) => handleChange('', 'name', e.target.value)}
        />
      </Grid>
      
      <Grid item xs={12} md={6}>
        <FormControl fullWidth>
          <InputLabel>Workflow Type</InputLabel>
          <Select
            value={formData.workflow_type}
            onChange={(e) => handleChange('', 'workflow_type', e.target.value)}
          >
            <MenuItem value="full_workflow">Full Workflow (DFT → ML → MD → Analysis)</MenuItem>
            <MenuItem value="dft_only">DFT Only</MenuItem>
            <MenuItem value="ml_training">ML Training Only</MenuItem>
            <MenuItem value="md_simulation">MD Simulation Only</MenuItem>
          </Select>
        </FormControl>
      </Grid>
      
      <Grid item xs={12}>
        <TextField
          fullWidth
          multiline
          rows={3}
          label="Description (optional)"
          value={formData.description}
          onChange={(e) => handleChange('', 'description', e.target.value)}
        />
      </Grid>
    </Grid>
  );
  
  const renderDFTStep = () => (
    <Grid container spacing={3}>
      <Grid item xs={12}>
        <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
          <Typography variant="h6">DFT Calculation Settings</Typography>
          
          <Box sx={{ display: 'flex', gap: 1 }}>
            {['fast', 'balanced', 'accurate'].map((preset) => (
              <Chip
                key={preset}
                label={preset.charAt(0).toUpperCase() + preset.slice(1)}
                onClick={() => applyPreset(preset)}
                clickable
                color="primary"
                variant="outlined"
              />
            ))}
          </Box>
        </Box>
      </Grid>
      
      <Grid item xs={12} md={6}>
        <FormControl fullWidth>
          <InputLabel>DFT Code</InputLabel>
          <Select
            value={formData.dft_config.code}
            onChange={(e) => handleChange('dft_config', 'code', e.target.value)}
          >
            <MenuItem value="vasp">VASP</MenuItem>
            <MenuItem value="espresso">Quantum ESPRESSO</MenuItem>
            <MenuItem value="abacus">ABACUS</MenuItem>
          </Select>
        </FormControl>
      </Grid>
      
      <Grid item xs={12} md={6}>
        <FormControl fullWidth>
          <InputLabel>Functional</InputLabel>
          <Select
            value={formData.dft_config.functional}
            onChange={(e) => handleChange('dft_config', 'functional', e.target.value)}
          >
            <MenuItem value="PBE">PBE</MenuItem>
            <MenuItem value="PBEsol">PBEsol</MenuItem>
            <MenuItem value="SCAN">SCAN</MenuItem>
            <MenuItem value="HSE06">HSE06</MenuItem>
          </Select>
        </FormControl>
      </Grid>
      
      <Grid item xs={12} md={6}>
        <Typography gutterBottom>Energy Cutoff (eV): {formData.dft_config.encut}</Typography>
        <Slider
          value={formData.dft_config.encut}
          onChange={(e, value) => handleChange('dft_config', 'encut', value)}
          min={200}
          max={1000}
          step={20}
          marks={[{ value: 400, label: '400' }, { value: 520, label: '520' }, { value: 700, label: '700' }]}
          valueLabelDisplay="auto"
        />
      </Grid>
      
      <Grid item xs={12} md={6}>
        <TextField
          fullWidth
          type="number"
          label="Number of CPU Cores"
          value={formData.dft_config.ncores}
          onChange={(e) => handleChange('dft_config', 'ncores', parseInt(e.target.value))}
          inputProps={{ min: 1, max: 128 }}
        />
      </Grid>
      
      <Grid item xs={12}>
        <FormControlLabel
          control={
            <Switch
              checked={formData.skip_dft}
              onChange={(e) => handleChange('', 'skip_dft', e.target.checked)}
            />
          }
          label="Skip DFT (use existing data)"
        />
      </Grid>
    </Grid>
  );
  
  const renderMLStep = () => (
    <Grid container spacing={3}>
      <Grid item xs={12}>
        <Typography variant="h6">ML Potential Training</Typography>
      </Grid>
      
      <Grid item xs={12} md={6}>
        <FormControl fullWidth>
          <InputLabel>Framework</InputLabel>
          <Select
            value={formData.ml_config.framework}
            onChange={(e) => handleChange('ml_config', 'framework', e.target.value)}
          >
            <MenuItem value="deepmd">DeepMD</MenuItem>
            <MenuItem value="nep">NEP (GPUMD)</MenuItem>
            <MenuItem value="mace">MACE</MenuItem>
          </Select>
        </FormControl>
      </Grid>
      
      <Grid item xs={12} md={6}>
        <FormControl fullWidth>
          <InputLabel>Training Preset</InputLabel>
          <Select
            value={formData.ml_config.preset}
            onChange={(e) => handleChange('ml_config', 'preset', e.target.value)}
          >
            <MenuItem value="fast">Fast (lower accuracy)</MenuItem>
            <MenuItem value="balanced">Balanced</MenuItem>
            <MenuItem value="accurate">Accurate (slower)</MenuItem>
          </Select>
        </FormControl>
      </Grid>
      
      <Grid item xs={12} md={6}>
        <TextField
          fullWidth
          type="number"
          label="Number of Ensemble Models"
          value={formData.ml_config.num_models}
          onChange={(e) => handleChange('ml_config', 'num_models', parseInt(e.target.value))}
          helperText="More models = better uncertainty quantification"
          inputProps={{ min: 1, max: 10 }}
        />
      </Grid>
      
      <Grid item xs={12}>
        <FormControlLabel
          control={
            <Switch
              checked={formData.skip_ml}
              onChange={(e) => handleChange('', 'skip_ml', e.target.checked)}
            />
          }
          label="Skip ML training (use existing model)"
        />
      </Grid>
    </Grid>
  );
  
  const renderMDStep = () => (
    <Grid container spacing={3}>
      <Grid item xs={12}>
        <Typography variant="h6">Molecular Dynamics Simulation</Typography>
      </Grid>
      
      <Grid item xs={12} md={6}>
        <FormControl fullWidth>
          <InputLabel>Ensemble</InputLabel>
          <Select
            value={formData.md_config.ensemble}
            onChange={(e) => handleChange('md_config', 'ensemble', e.target.value)}
          >
            <MenuItem value="nvt">NVT (Constant T, V)</MenuItem>
            <MenuItem value="npt">NPT (Constant T, P)</MenuItem>
            <MenuItem value="nve">NVE (Microcanonical)</MenuItem>
          </Select>
        </FormControl>
      </Grid>
      
      <Grid item xs={12}>
        <Typography gutterBottom>Temperatures (K)</Typography>
        <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 1, mb: 2 }}>
          {[300, 400, 500, 600, 700, 800, 900, 1000].map((temp) => (
            <Chip
              key={temp}
              label={`${temp}K`}
              color={formData.md_config.temperatures.includes(temp) ? 'primary' : 'default'}
              onClick={() => {
                const newTemps = formData.md_config.temperatures.includes(temp)
                  ? formData.md_config.temperatures.filter((t) => t !== temp)
                  : [...formData.md_config.temperatures, temp].sort((a, b) => a - b);
                handleChange('md_config', 'temperatures', newTemps);
              }}
            />
          ))}
        </Box>
      </Grid>
      
      <Grid item xs={12} md={6}>
        <Typography gutterBottom>Production Steps: {formData.md_config.nsteps_prod.toLocaleString()}</Typography>
        <Slider
          value={formData.md_config.nsteps_prod}
          onChange={(e, value) => handleChange('md_config', 'nsteps_prod', value)}
          min={10000}
          max={500000}
          step={10000}
          marks={[
            { value: 100000, label: '100K' },
            { value: 300000, label: '300K' },
            { value: 500000, label: '500K' },
          ]}
          valueLabelDisplay="auto"
        />
      </Grid>
      
      <Grid item xs={12} md={6}>
        <TextField
          fullWidth
          type="number"
          label="Timestep (fs)"
          value={formData.md_config.timestep}
          onChange={(e) => handleChange('md_config', 'timestep', parseFloat(e.target.value))}
          inputProps={{ min: 0.1, max: 5, step: 0.1 }}
        />
      </Grid>
      
      <Grid item xs={12}>
        <Typography variant="subtitle1" gutterBottom>Analysis Options</Typography>
        <FormControlLabel
          control={
            <Switch
              checked={formData.analysis_config.compute_diffusion}
              onChange={(e) => handleChange('analysis_config', 'compute_diffusion', e.target.checked)}
            />
          }
          label="Compute diffusion coefficients"
        />
        <FormControlLabel
          control={
            <Switch
              checked={formData.analysis_config.compute_conductivity}
              onChange={(e) => handleChange('analysis_config', 'compute_conductivity', e.target.checked)}
            />
          }
          label="Compute ionic conductivity"
        />
        <FormControlLabel
          control={
            <Switch
              checked={formData.analysis_config.compute_activation_energy}
              onChange={(e) => handleChange('analysis_config', 'compute_activation_energy', e.target.checked)}
            />
          }
          label="Compute activation energy"
        />
      </Grid>
    </Grid>
  );
  
  const getStepContent = (step) => {
    switch (step) {
      case 0:
        return renderStructureStep();
      case 1:
        return renderDFTStep();
      case 2:
        return renderMLStep();
      case 3:
        return renderMDStep();
      default:
        return 'Unknown step';
    }
  };
  
  return (
    <Box sx={{ maxWidth: 900, mx: 'auto' }}>
      <Typography variant="h4" gutterBottom>
        Submit New Task
      </Typography>
      
      {(error || submitError) && (
        <Alert severity="error" sx={{ mb: 3 }}>{error || submitError}</Alert>
      )}
      
      <Paper sx={{ p: 4 }}>
        <Stepper activeStep={activeStep} sx={{ mb: 4 }}>
          {steps.map((label) => (
            <Step key={label}>
              <StepLabel>{label}</StepLabel>
            </Step>
          ))}
        </Stepper>
        
        <Box sx={{ mt: 2, mb: 4 }}>{getStepContent(activeStep)}</Box>
        
        <Box sx={{ display: 'flex', justifyContent: 'space-between', mt: 4 }}>
          <Button
            disabled={activeStep === 0}
            onClick={handleBack}
            variant="outlined"
          >
            Back
          </Button>
          
          <Box>
            {activeStep === steps.length - 1 ? (
              <Button
                variant="contained"
                onClick={handleSubmit}
                disabled={isLoading || !formData.name}
                startIcon={isLoading ? <CircularProgress size={20} /> : null}
              >
                {isLoading ? 'Submitting...' : 'Submit Task'}
              </Button>
            ) : (
              <Button
                variant="contained"
                onClick={handleNext}
                disabled={activeStep === 0 && !formData.material_id && !formData.formula}
              >
                Next
              </Button>
            )}
          </Box>
        </Box>
      </Paper>
    </Box>
  );
};

export default TaskSubmission;
