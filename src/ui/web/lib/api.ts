import axios from 'axios';

const API_BASE_URL = 'http://localhost:8000';

export const api = axios.create({
    baseURL: API_BASE_URL,
    headers: {
        'Content-Type': 'application/json',
    },
});

export interface TaskRequest {
    query: string;
}

export interface TaskResponse {
    task_id: string;
    task_type: string;
    description: string;
    status: string;
}

export const AgentService = {
    createTask: async (query: string): Promise<TaskResponse> => {
        const res = await api.post<TaskResponse>('/tasks/create', { query });
        return res.data;
    },

    chat: async (message: string): Promise<any> => {
        const res = await api.post('/tasks/chat', { message });
        return res.data;
    },

    getTaskStatus: async (taskId: string) => {
        const res = await api.get(`/tasks/${taskId}`);
        return res.data;
    },

    executeTask: async (taskId: string) => {
        const res = await api.post(`/tasks/execute/${taskId}`);
        return res.data;
    },

    getSystemStatus: async () => {
        // Health check from main
        const res = await api.get('/health');
        return res.data;
    },

    getTheoryStatus: async () => {
        const res = await api.get('/theory/status');
        return res.data;
    },

    getMaterials: async () => {
        const res = await api.get('/theory/materials');
        return res.data;
    },

    uploadExperiment: async (file: File, method: string = "lsv"): Promise<ExperimentResponse> => {
        const formData = new FormData();
        formData.append('file', file);

        const res = await api.post<ExperimentResponse>(`/experiment/upload?method=${method}`, formData, {
            headers: {
                'Content-Type': 'multipart/form-data',
            },
        });
        return res.data;
    },

    getMaterialDetails: async (material_id: string) => {
        const res = await api.get(`/theory/materials/${material_id}`);
        return res.data;
    }
};

export interface Material {
    id: number;
    material_id: string;
    formula: string;
    formation_energy: number;
    cif_path: string;
    cif_content?: string;
}

export interface ExperimentResponse {
    message: string;
    filename: string;
    analysis: any; // Flexible for LSV/CV results
}
