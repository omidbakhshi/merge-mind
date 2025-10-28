import axios, { AxiosResponse } from 'axios';

const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:8080';
const API_TOKEN = process.env.REACT_APP_API_TOKEN || '';

const api = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
    ...(API_TOKEN && { 'Authorization': `Bearer ${API_TOKEN}` })
  }
});

// Request interceptor for error handling
api.interceptors.response.use(
  (response) => response,
  (error) => {
    if (error.response?.status === 401) {
      console.error('Authentication failed. Please check your API token.');
    } else if (error.response?.status >= 500) {
      console.error('Server error. Please try again later.');
    }
    return Promise.reject(error);
  }
);

export interface Project {
  project_id: number;
  name: string;
  description?: string;
  review_enabled: boolean;
  review_drafts: boolean;
  min_lines_changed: number;
  max_files_per_review: number;
  excluded_paths: string[];
  included_extensions: string[];
  custom_prompts?: Record<string, string>;
  review_model?: string;
  team_preferences?: string[];
}

export interface HealthCheck {
  status: string;
  timestamp: string;
  version: string;
  uptime_seconds: number;
}

export interface Stats {
  projects_configured: number;
  active_reviews: number;
  learning_stats: Record<string, any>;
  cache_size: number;
}

export interface ReviewHistory {
  reviews: Array<{
    id: string;
    project_id: number;
    mr_iid: number;
    status: string;
    created_at: string;
    completed_at?: string;
    results_count: number;
  }>;
}

export interface ReloadResponse {
  success: boolean;
  message: string;
  changes: Record<string, { old: any; new: any }>;
}

class ApiService {
  // Health & Status
  async getHealth(): Promise<HealthCheck> {
    const response = await api.get('/health');
    return response.data;
  }

  async getDetailedHealth(): Promise<any> {
    const response = await api.get('/health/detailed');
    return response.data;
  }

  async getStats(): Promise<Stats> {
    const response = await api.get('/stats');
    return response.data;
  }

  async getMetrics(): Promise<any> {
    const response = await api.get('/metrics');
    return response.data;
  }

  // Project Management
  async getProjects(): Promise<{ projects: Project[] }> {
    const response = await api.get('/projects');
    return response.data;
  }

  async getProject(projectId: number): Promise<Project> {
    const response = await api.get(`/projects/${projectId}`);
    return response.data;
  }

  async createProject(project: Omit<Project, 'project_id'> & { project_id: number }): Promise<Project> {
    const response = await api.post('/projects', project);
    return response.data;
  }

  async updateProject(projectId: number, updates: Partial<Project>): Promise<Project> {
    const response = await api.put(`/projects/${projectId}`, updates);
    return response.data;
  }

  async deleteProject(projectId: number): Promise<void> {
    await api.delete(`/projects/${projectId}`);
  }

  // Review Management
  async getActiveReviews(): Promise<any> {
    const response = await api.get('/reviews/active');
    return response.data;
  }

  async getReviewHistory(projectId?: number, limit: number = 10): Promise<ReviewHistory> {
    const params = new URLSearchParams();
    if (projectId) params.append('project_id', projectId.toString());
    params.append('limit', limit.toString());

    const response = await api.get(`/reviews/history?${params}`);
    return response.data;
  }

  // Configuration Management
  async reloadConfig(): Promise<ReloadResponse> {
    const response = await api.post('/reload');
    return response.data;
  }

  async getConfig(): Promise<any> {
    const response = await api.get('/config');
    return response.data;
  }
}

export const apiService = new ApiService();
export default api;