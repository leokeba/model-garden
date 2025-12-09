// Use relative URL for API - works with any host (localhost, remote IP, domain)
const API_BASE = '/api/v1';

interface Model {
  id: string;
  name: string;
  base_model: string;
  status: string;
  created_at: string;
  updated_at: string;
  size_bytes?: number;
  path: string;
  training_job_id?: string;
  config?: any;
  metrics?: any;
  // Optional fields that may not exist in the API but are used in the UI
  type?: string;
  size?: number;
  files?: Array<{
    name: string;
    size: number;
    modified: string;
  }>;
  // File existence check (added by backend)
  file_exists?: boolean;
  file_count?: number;
  // HuggingFace Hub info (if uploaded)
  hub_url?: string;
  hub_repo_id?: string;
}

interface TrainingJob {
  id: string;
  job_id?: string;
  name: string;
  status: string;
  base_model: string;
  dataset_path: string;
  validation_dataset_path?: string;
  output_dir: string;
  created_at: string;
  started_at?: string;
  completed_at?: string;
  progress?: {
    current_step: number;
    total_steps: number;
    epoch: number;
    percentage?: number;
    eta_seconds?: number;
    steps_per_second?: number;
  };
  current_step?: number;
  total_steps?: number;
  current_epoch?: number;
  error_message?: string;
  hyperparameters?: any;
  lora_config?: any;
  model_type?: string;
  save_method?: string;
  backend?: string;  // Training backend (unsloth, transformers, etc.)
  // Dataset stats
  dataset_size?: number;
  dataset_num_samples?: number;
  // Quality settings
  quality_mode?: boolean;
  load_in_16bit?: boolean;
  load_in_8bit?: boolean;
  // Early stopping
  early_stopping_enabled?: boolean;
  early_stopping_patience?: number;
  early_stopping_threshold?: number;
  // Selective loss
  selective_loss?: boolean;
  selective_loss_level?: string;
  selective_loss_schema_keys?: string[];
  selective_loss_masking_strategy?: string;
  selective_loss_structural_weight?: number;
  selective_loss_masking_start_epoch?: number;
  selective_loss_mask_every_n_steps?: number;
  selective_loss_mask_for_n_steps?: number;
  selective_loss_verbose?: boolean;
  // Rerun metadata
  rerun_from?: string;
  rerun_from_name?: string;
  config?: {
    name: string;
    base_model: string;
    dataset_path: string;
    output_dir: string;
    hyperparameters: any;
    lora_config: any;
    model_type?: string;
  };
  metrics?: {
    training?: Array<{
      step: number;
      loss: number;
      learning_rate?: number;
    }>;
    validation?: Array<{
      step: number;
      loss: number;
    }>;
  };
  logs?: string[];
}

interface PaginatedResponse<T> {
  items: T[];
  models?: T[]; // For backward compatibility
  total: number;
  page: number;
  page_size: number;
  pages: number;
}

interface SystemStatus {
  system: {
    cpu_count: number;
    cpu_percent?: number;
    memory_total: number;
    memory_available: number;
    memory_used: number;
    memory_percent: number;
    disk_usage: {
      total: number;
      used: number;
      free: number;
      percent: number;
    };
  };
  gpu: {
    available: boolean;
    device_count?: number;
    devices?: Array<{
      id: number;
      name: string;
      memory: {
        total: number;
        used: number;
        free: number;
        used_percent: number;
      };
      utilization: {
        gpu: number | null;
        memory: number | null;
      };
      temperature: number | null;
      power: {
        usage: number;
        limit: number;
      } | null;
    }>;
    // Backward compatibility fields
    current_device?: number;
    device_name?: string;
    memory_allocated?: number;
    memory_reserved?: number;
  };
  storage: {
    models_count: number;
    training_jobs_count: number;
    active_jobs: number;
  };
}

// Model Registry Types
interface RegistryModelRequirements {
  min_vram_gb: number;
  recommended_vram_gb: number;
  gpu_required: boolean;
  min_gpu_compute_capability?: string;
}

interface RegistryModelCapabilities {
  supports_vision: boolean;
  supports_chat: boolean;
  supports_function_calling: boolean;
  supports_structured_output: boolean;
  max_sequence_length: number;
  context_window?: number;
}

interface RegistryHyperparametersDefaults {
  learning_rate: number;
  num_epochs: number;
  batch_size: number;
  gradient_accumulation_steps: number;
  max_steps: number;
  warmup_steps: number;
  logging_steps: number;
  save_steps: number;
  eval_steps?: number | null;
  optim: string;
  weight_decay: number;
  lr_scheduler_type: string;
  max_grad_norm: number;
  adam_beta1?: number;
  adam_beta2?: number;
  adam_epsilon?: number;
  dataloader_num_workers?: number;
  dataloader_pin_memory?: boolean;
  eval_strategy?: string;
  load_best_model_at_end?: boolean;
  metric_for_best_model?: string;
  save_total_limit?: number;
}

interface RegistryLoRADefaults {
  r: number;
  lora_alpha: number;
  lora_dropout: number;
  lora_bias: string;
  use_rslora: boolean;
  use_gradient_checkpointing: string;
  random_state: number;
  target_modules: string[] | null;
  task_type: string;
}

interface RegistryInferenceDefaults {
  tensor_parallel_size: number;
  gpu_memory_utilization: number;
  max_model_len?: number | null;
  dtype: string;
  quantization?: string | null;
}

interface RegistryModelInfo {
  id: string;
  name: string;
  description: string;
  parameters: string;
  category: string;
  tags: string[];
  recommended_for?: string[];
  requirements: RegistryModelRequirements;
  capabilities: RegistryModelCapabilities;
  training_defaults: {
    hyperparameters: RegistryHyperparametersDefaults;
    lora_config: RegistryLoRADefaults;
    save_method?: string;
  };
  inference_defaults: RegistryInferenceDefaults;
  version?: string;
  source_url?: string;
  license?: string;
  notes?: string;
  status?: string;
  is_vision?: boolean;
  is_quantized?: boolean;
  min_vram_gb?: number;
  recommended_vram_gb?: number;
}

interface RegistryCategory {
  id: string;
  name: string;
  description: string;
}

interface RegistryModelsResponse {
  models: RegistryModelInfo[];
  total: number;
  category?: string;
}

// Training Backend Types
interface TrainingBackend {
  name: string;
  description: string;
  supports_text: boolean;
  supports_vision: boolean;
}

interface Config {
  hf_user: string;
  models_dir: string;
  hf_home: string;
}

// Settings Types
interface UnslothStatus {
  installed: boolean;
  version: string | null;
  description: string;
}

interface PackageOperationStatus {
  in_progress: boolean;
  operation: string | null;
  output: string[];
  success: boolean | null;
  error: string | null;
}

// GPU Memory Profile Types
interface GPUMemoryBreakdown {
  weights_gb: number;
  kv_cache_gb: number;
  cuda_graphs_gb: number;
  other_gb: number;
  total_used_gb: number;
  available_gb: number;
}

interface GPUMemoryUtilization {
  used_percent: number;
  available_percent: number;
}

interface GPUMemoryConfig {
  model_path: string;
  max_model_len: number;
  enforce_eager: boolean;
  tensor_parallel_size: number;
  gpu_memory_utilization: number;
}

interface GPUMemoryKVCache {
  tokens: number;
  max_concurrency: number;
}

interface GPUMemoryTiming {
  load_time_seconds: number;
}

interface GPUMemoryProfile {
  gpu: {
    name: string;
    total_memory_gb: number;
  };
  breakdown: GPUMemoryBreakdown;
  utilization: GPUMemoryUtilization;
  config: GPUMemoryConfig;
  kv_cache: GPUMemoryKVCache;
  timing: GPUMemoryTiming;
  weight_file_size_gb: number;
}

interface GPULiveStats {
  gpu_name: string;
  total_gb: number;
  allocated_gb: number;
  reserved_gb: number;
  free_gb: number;
  peak_allocated_gb: number;
  peak_reserved_gb: number;
  utilization_percent: number;
  error?: string;
}

interface GPUMemoryStats {
  live: GPULiveStats;
  profile: GPUMemoryProfile | null;
}

interface SystemSettings {
  optional_dependencies: {
    unsloth: UnslothStatus;
  };
  environment: {
    python_version: string;
    transformers_version: string;
    project_root: string;
  };
  service: {
    is_systemd_service: boolean;
    can_restart_service: boolean;
  };
  package_operation: PackageOperationStatus;
}

class APIClient {
  private baseURL: string;

  constructor(baseURL: string) {
    this.baseURL = baseURL;
  }

  private async request<T>(
    endpoint: string,
    options?: RequestInit
  ): Promise<T> {
    const response = await fetch(`${this.baseURL}${endpoint}`, {
      ...options,
      headers: {
        'Content-Type': 'application/json',
        ...options?.headers,
      },
    });

    if (!response.ok) {
      const error = await response.text();
      throw new Error(`API Error: ${response.statusText} - ${error}`);
    }

    return response.json();
  }

  // Configuration
  async getConfig(): Promise<Config> {
    const response = await this.request<{ success: boolean; data: Config }>('/config');
    return response.data;
  }

  // Models
  async getModels(): Promise<PaginatedResponse<Model>> {
    return this.request('/models');
  }

  async getModel(id: string): Promise<Model> {
    return this.request(`/models/${id}`);
  }

  async deleteModel(id: string): Promise<{ success: boolean; message: string }> {
    return this.request(`/models/${id}`, { method: 'DELETE' });
  }

  async uploadModelToHub(
    modelId: string,
    params: {
      repo_id: string;
      private?: boolean;
      commit_message?: string;
      repo_description?: string;
    }
  ): Promise<{ success: boolean; message: string; repo_id: string; url: string; commit_url: string }> {
    return this.request(`/models/${modelId}/upload-to-hub`, {
      method: 'POST',
      body: JSON.stringify(params),
    });
  }

  async renameModel(modelId: string, newName: string): Promise<{ success: boolean; data: { old_id: string; new_id: string }; message: string }> {
    return this.request(`/models/${encodeURIComponent(modelId)}/rename`, {
      method: 'POST',
      body: JSON.stringify({ new_name: newName }),
    });
  }

  async generateText(modelName: string, params: {
    prompt: string;
    max_length?: number;
    temperature?: number;
    top_p?: number;
    top_k?: number;
  }): Promise<{ success: boolean; data: { generated_text: string }; message: string }> {
    return this.request(`/models/${modelName}/generate`, {
      method: 'POST',
      body: JSON.stringify(params),
    });
  }

  // Training
  async createTrainingJob(data: any): Promise<{ success: boolean; data: { job_id: string }; message: string }> {
    return this.request('/training/jobs', {
      method: 'POST',
      body: JSON.stringify(data),
    });
  }

  async getTrainingJobs(): Promise<PaginatedResponse<TrainingJob>> {
    return this.request('/training/jobs');
  }

  async getTrainingJob(id: string): Promise<TrainingJob> {
    return this.request(`/training/jobs/${id}`);
  }

  async cancelTrainingJob(id: string): Promise<{ success: boolean; message: string }> {
    return this.request(`/training/jobs/${id}`, {
      method: 'DELETE',
    });
  }

  async rerunTrainingJob(id: string): Promise<{ success: boolean; data: { job_id: string; original_job_id: string; queue_position: number | null; name: string }; message: string }> {
    return this.request(`/training/jobs/${id}/rerun`, {
      method: 'POST',
    });
  }

  // System
  async getSystemStatus(): Promise<SystemStatus> {
    return this.request('/system/status');
  }

  // Health check
  async getHealth(): Promise<{ status: string; message: string }> {
    const response = await fetch(`${this.baseURL.replace('/api/v1', '')}/health`);
    return response.json();
  }

  // Inference
  async getInferenceStatus(): Promise<{
    loaded: boolean;
    loading?: {
      job_id: string;
      model_path: string;
      started_at: string;
      status_message: string;
    };
    queue?: {
      count: number;
      jobs: Array<{
        job_id: string;
        model_path: string;
        queued_at: string;
        position: number;
      }>;
    };
    model_info: {
      model_path: string;
      is_loaded: boolean;
      tensor_parallel_size: number;
      gpu_memory_utilization: number;
      max_model_len: number | null;
      max_num_seqs?: number;
      enforce_eager?: boolean;
      dtype: string;
      quantization: string | null;
      // LoRA adapter info
      is_lora_adapter?: boolean;
      base_model?: string;
      adapter_path?: string;
      lora_enabled?: boolean;
      is_vision_adapter?: boolean;
      merged_automatically?: boolean;
      note?: string;
    } | null;
  }> {
    return this.request('/inference/status');
  }

  async loadModel(params: {
    model_path: string;
    tensor_parallel_size?: number;
    gpu_memory_utilization?: number;
    max_model_len?: number | null;
    max_num_seqs?: number | null;
    enforce_eager?: boolean | null;
    limit_mm_per_prompt?: Record<string, number> | null;
    dtype?: string;
    quantization?: string | null;
  }): Promise<{ success: boolean; message: string; data?: any }> {
    const body: Record<string, any> = {
      model_path: params.model_path,
      tensor_parallel_size: params.tensor_parallel_size ?? 1,
      gpu_memory_utilization: params.gpu_memory_utilization ?? 0.0,  // Use ?? to allow 0
      dtype: params.dtype || 'auto',
    };

    // Only include optional params if explicitly set
    if (params.max_model_len !== undefined && params.max_model_len !== null) {
      body.max_model_len = params.max_model_len;
    }
    if (params.max_num_seqs !== undefined && params.max_num_seqs !== null) {
      body.max_num_seqs = params.max_num_seqs;
    }
    if (params.enforce_eager !== undefined && params.enforce_eager !== null) {
      body.enforce_eager = params.enforce_eager;
    }
    if (params.limit_mm_per_prompt !== undefined && params.limit_mm_per_prompt !== null) {
      body.limit_mm_per_prompt = params.limit_mm_per_prompt;
    }
    if (params.quantization !== undefined && params.quantization !== null) {
      body.quantization = params.quantization;
    }

    return this.request('/inference/load', {
      method: 'POST',
      body: JSON.stringify(body),
    });
  }

  async reloadModel(): Promise<{ success: boolean; message: string; data?: any }> {
    // Get current model info, unload, then reload with same settings
    const status = await this.getInferenceStatus();
    if (!status.loaded || !status.model_info) {
      return { success: false, message: 'No model currently loaded to reload' };
    }

    const modelInfo = status.model_info;
    await this.unloadModel();

    return this.loadModel({
      model_path: modelInfo.model_path,
      tensor_parallel_size: modelInfo.tensor_parallel_size,
      gpu_memory_utilization: modelInfo.gpu_memory_utilization,
      max_model_len: modelInfo.max_model_len,
      dtype: modelInfo.dtype,
      quantization: modelInfo.quantization,
    });
  }

  async unloadModel(): Promise<{ success: boolean; message: string }> {
    return this.request('/inference/unload', {
      method: 'POST',
    });
  }

  // Model Registry
  async getRegistryModels(category?: string): Promise<RegistryModelsResponse> {
    const endpoint = category ? `/registry/models?category=${category}` : '/registry/models';
    const response = await this.request<{ success: boolean; data: RegistryModelInfo[]; total: number }>(endpoint);
    return {
      models: response.data,
      total: response.total,
      category,
    };
  }

  async getRegistryModel(id: string): Promise<RegistryModelInfo> {
    const response = await this.request<{ success: boolean; data: RegistryModelInfo }>(`/registry/models/${encodeURIComponent(id)}`);
    return response.data;
  }

  async getRegistryCategories(): Promise<{ categories: Record<string, RegistryCategory> }> {
    const response = await this.request<{ success: boolean; data: Record<string, RegistryCategory> }>('/registry/categories');
    return { categories: response.data };
  }

  async validateModelForTraining(modelId: string, config: any): Promise<{
    valid: boolean;
    errors: string[];
    warnings: string[];
  }> {
    return this.request('/registry/validate/training', {
      method: 'POST',
      body: JSON.stringify({ model_id: modelId, config }),
    });
  }

  async validateModelForInference(modelId: string, config: any): Promise<{
    valid: boolean;
    errors: string[];
    warnings: string[];
  }> {
    return this.request('/registry/validate/inference', {
      method: 'POST',
      body: JSON.stringify({ model_id: modelId, config }),
    });
  }

  // Training Backends
  async getBackends(): Promise<{ backends: TrainingBackend[]; total: number }> {
    const response = await this.request<{ success: boolean; data: TrainingBackend[]; total: number }>('/system/backends');
    return {
      backends: response.data,
      total: response.total,
    };
  }

  // System Settings
  async getSettings(): Promise<SystemSettings> {
    const response = await this.request<{ success: boolean; data: SystemSettings }>('/system/settings');
    return response.data;
  }

  async getGpuMemoryStats(): Promise<GPUMemoryStats> {
    const response = await this.request<{ success: boolean; data: GPUMemoryStats }>('/system/gpu/memory');
    return response.data;
  }

  async getGpuMemoryProfile(): Promise<GPUMemoryProfile | null> {
    const response = await this.request<{ success: boolean; data: GPUMemoryProfile | null }>('/system/gpu/memory/profile');
    return response.data;
  }

  async installUnsloth(): Promise<{ success: boolean; message: string; data?: { operation: string; command: string } }> {
    return this.request('/system/unsloth/install', { method: 'POST' });
  }

  async uninstallUnsloth(): Promise<{ success: boolean; message: string; data?: { operation: string; command: string } }> {
    return this.request('/system/unsloth/uninstall', { method: 'POST' });
  }

  async getUnslothOperationStatus(): Promise<{ success: boolean; data: PackageOperationStatus }> {
    return this.request('/system/unsloth/status');
  }

  async restartService(): Promise<{ success: boolean; message: string }> {
    return this.request('/system/restart', { method: 'POST' });
  }

  // Generic methods for other endpoints
  async get<T = any>(endpoint: string): Promise<T> {
    return this.request(endpoint);
  }

  async post<T = any>(endpoint: string, data?: any): Promise<T> {
    return this.request(endpoint, {
      method: 'POST',
      body: JSON.stringify(data),
    });
  }

  async put<T = any>(endpoint: string, data?: any): Promise<T> {
    return this.request(endpoint, {
      method: 'PUT',
      body: JSON.stringify(data),
    });
  }

  async delete<T = any>(endpoint: string): Promise<T> {
    return this.request(endpoint, {
      method: 'DELETE',
    });
  }
}

export const api = new APIClient(API_BASE);
export type {
  Config, GPULiveStats, GPUMemoryBreakdown, GPUMemoryConfig, GPUMemoryKVCache, GPUMemoryProfile, GPUMemoryStats, GPUMemoryTiming, GPUMemoryUtilization, Model, PackageOperationStatus, RegistryCategory, RegistryHyperparametersDefaults, RegistryInferenceDefaults, RegistryLoRADefaults, RegistryModelCapabilities, RegistryModelInfo, RegistryModelRequirements, RegistryModelsResponse, SystemSettings, SystemStatus,
  TrainingBackend,
  TrainingJob, UnslothStatus
};

