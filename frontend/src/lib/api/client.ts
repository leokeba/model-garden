const API_BASE = 'http://localhost:8000/api/v1';

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
}

interface TrainingJob {
  id: string;
  job_id?: string;
  name: string;
  status: string;
  base_model: string;
  dataset_path: string;
  output_dir: string;
  created_at: string;
  started_at?: string;
  completed_at?: string;
  progress?: {
    current_step: number;
    total_steps: number;
    epoch: number;
  };
  current_step?: number;
  total_steps?: number;
  current_epoch?: number;
  error_message?: string;
  hyperparameters?: any;
  lora_config?: any;
  config?: {
    name: string;
    base_model: string;
    dataset_path: string;
    output_dir: string;
    hyperparameters: any;
    lora_config: any;
  };
  logs?: string[];
}

interface PaginatedResponse<T> {
  items: T[];
  total: number;
  page: number;
  page_size: number;
  pages: number;
}

interface SystemStatus {
  system: {
    cpu_count: number;
    memory_total: number;
    memory_available: number;
    disk_usage: {
      total: number;
      used: number;
      free: number;
    };
  };
  gpu: {
    available: boolean;
    device_count?: number;
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
  
  // System
  async getSystemStatus(): Promise<SystemStatus> {
    return this.request('/system/status');
  }
  
  // Health check
  async getHealth(): Promise<{ status: string; message: string }> {
    const response = await fetch(`${this.baseURL.replace('/api/v1', '')}/health`);
    return response.json();
  }
}

export const api = new APIClient(API_BASE);
export type { Model, TrainingJob, SystemStatus };