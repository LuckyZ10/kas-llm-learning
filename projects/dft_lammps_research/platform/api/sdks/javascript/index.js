/**
 * DFT+LAMMPS JavaScript SDK
 * 
 * Official JavaScript/TypeScript client for the DFT+LAMMPS API Platform.
 * 
 * Installation:
 *   npm install dft-lammps-client
 * 
 * Quick Start:
 *   const { Client } = require('dft-lammps-client');
 *   const client = new Client({ apiKey: 'your-api-key' });
 *   const project = await client.projects.create({ name: 'My Project' });
 */

const axios = require('axios');
const crypto = require('crypto');

const DEFAULT_BASE_URL = 'https://api.dft-lammps.org';
const DEFAULT_TIMEOUT = 30000;
const DEFAULT_MAX_RETRIES = 3;

// Custom error classes
class DFTLAMMPSError extends Error {
  constructor(message, statusCode, response) {
    super(message);
    this.name = 'DFTLAMMPSError';
    this.statusCode = statusCode;
    this.response = response;
  }
}

class AuthenticationError extends DFTLAMMPSError {
  constructor(message, response) {
    super(message, 401, response);
    this.name = 'AuthenticationError';
  }
}

class RateLimitError extends DFTLAMMPSError {
  constructor(message, response) {
    super(message, 429, response);
    this.name = 'RateLimitError';
  }
}

class NotFoundError extends DFTLAMMPSError {
  constructor(message, response) {
    super(message, 404, response);
    this.name = 'NotFoundError';
  }
}

class ValidationError extends DFTLAMMPSError {
  constructor(message, response) {
    super(message, 400, response);
    this.name = 'ValidationError';
  }
}

/**
 * HTTP Client with retries and authentication
 */
class HTTPClient {
  constructor(options = {}) {
    this.apiKey = options.apiKey || process.env.DFT_LAMMPS_API_KEY;
    if (!this.apiKey) {
      throw new AuthenticationError('API key required. Set DFT_LAMMPS_API_KEY env var or pass to constructor.');
    }

    this.baseURL = (options.baseURL || DEFAULT_BASE_URL).replace(/\/$/, '');
    this.timeout = options.timeout || DEFAULT_TIMEOUT;
    this.maxRetries = options.maxRetries || DEFAULT_MAX_RETRIES;

    this.client = axios.create({
      baseURL: this.baseURL,
      timeout: this.timeout,
      headers: {
        'Authorization': `Bearer ${this.apiKey}`,
        'Content-Type': 'application/json',
        'User-Agent': `dft-lammps-js/1.0.0`,
      },
    });

    // Request interceptor for logging
    this.client.interceptors.request.use(
      (config) => {
        config.metadata = { startTime: Date.now() };
        return config;
      },
      (error) => Promise.reject(error)
    );

    // Response interceptor for error handling
    this.client.interceptors.response.use(
      (response) => response,
      async (error) => {
        const config = error.config;

        // Retry logic
        if (!config || !config.retry) {
          config.retry = 0;
        }

        if (config.retry < this.maxRetries && this._shouldRetry(error)) {
          config.retry += 1;
          const delay = Math.pow(2, config.retry) * 1000;
          await this._sleep(delay);
          return this.client(config);
        }

        return Promise.reject(this._handleError(error));
      }
    );
  }

  _shouldRetry(error) {
    if (!error.response) return true; // Network error
    const status = error.response.status;
    return status === 429 || (status >= 500 && status < 600);
  }

  _sleep(ms) {
    return new Promise(resolve => setTimeout(resolve, ms));
  }

  _handleError(error) {
    if (!error.response) {
      return new DFTLAMMPSError(error.message);
    }

    const status = error.response.status;
    const data = error.response.data;
    const message = data?.detail || data?.message || 'API error';

    switch (status) {
      case 400:
        return new ValidationError(message, data);
      case 401:
        return new AuthenticationError('Invalid API key', data);
      case 403:
        return new AuthenticationError('Permission denied', data);
      case 404:
        return new NotFoundError('Resource not found', data);
      case 429:
        return new RateLimitError('Rate limit exceeded. Try again later.', data);
      default:
        return new DFTLAMMPSError(message, status, data);
    }
  }

  async get(endpoint, params = {}) {
    const response = await this.client.get(endpoint, { params });
    return response.data;
  }

  async post(endpoint, data = null) {
    const response = await this.client.post(endpoint, data);
    return response.data;
  }

  async patch(endpoint, data = null) {
    const response = await this.client.patch(endpoint, data);
    return response.data;
  }

  async delete(endpoint) {
    const response = await this.client.delete(endpoint);
    return response.data;
  }
}

/**
 * Projects API
 */
class ProjectsAPI {
  constructor(client) {
    this.client = client;
  }

  /**
   * List projects with pagination
   * @param {Object} options - Query options
   * @returns {Promise<Object>} Paginated list of projects
   */
  async list(options = {}) {
    const params = {
      page: options.page || 1,
      page_size: options.pageSize || 20,
      ...(options.status && { status: options.status }),
      ...(options.projectType && { project_type: options.projectType }),
      ...(options.search && { search: options.search }),
    };

    return await this.client.get('/api/v1/projects', params);
  }

  /**
   * Create a new project
   * @param {Object} data - Project data
   * @returns {Promise<Object>} Created project
   */
  async create(data) {
    const payload = {
      name: data.name,
      description: data.description,
      project_type: data.projectType || 'battery_screening',
      target_properties: data.targetProperties || {},
      material_system: data.materialSystem,
      tags: data.tags || [],
    };

    return await this.client.post('/api/v1/projects', payload);
  }

  /**
   * Get project by ID
   * @param {string} projectId - Project ID
   * @returns {Promise<Object>} Project
   */
  async get(projectId) {
    return await this.client.get(`/api/v1/projects/${projectId}`);
  }

  /**
   * Update project
   * @param {string} projectId - Project ID
   * @param {Object} data - Update data
   * @returns {Promise<Object>} Updated project
   */
  async update(projectId, data) {
    return await this.client.patch(`/api/v1/projects/${projectId}`, data);
  }

  /**
   * Delete project
   * @param {string} projectId - Project ID
   * @returns {Promise<void>}
   */
  async delete(projectId) {
    await this.client.delete(`/api/v1/projects/${projectId}`);
  }
}

/**
 * Calculations API
 */
class CalculationsAPI {
  constructor(client) {
    this.client = client;
  }

  /**
   * Submit a calculation
   * @param {string} projectId - Project ID
   * @param {Object} data - Calculation data
   * @returns {Promise<Object>} Created calculation
   */
  async submit(projectId, data) {
    const payload = {
      structure: data.structure,
      calculation_type: data.calculationType || 'dft',
      parameters: data.parameters || {},
      priority: data.priority || 5,
    };

    return await this.client.post(`/api/v1/projects/${projectId}/calculations`, payload);
  }

  /**
   * Get calculation by ID
   * @param {string} calculationId - Calculation ID
   * @returns {Promise<Object>} Calculation
   */
  async get(calculationId) {
    return await this.client.get(`/api/v1/calculations/${calculationId}`);
  }

  /**
   * List calculations for a project
   * @param {string} projectId - Project ID
   * @param {Object} options - Query options
   * @returns {Promise<Array>} List of calculations
   */
  async list(projectId, options = {}) {
    const params = {
      page: options.page || 1,
      page_size: options.pageSize || 20,
      ...(options.status && { status: options.status }),
    };

    const response = await this.client.get(`/api/v1/projects/${projectId}/calculations`, params);
    return response.items || [];
  }

  /**
   * Wait for calculation to complete
   * @param {string} calculationId - Calculation ID
   * @param {Object} options - Wait options
   * @returns {Promise<Object>} Completed calculation
   */
  async wait(calculationId, options = {}) {
    const timeout = options.timeout || null;
    const pollInterval = options.pollInterval || 5000;
    const startTime = Date.now();

    while (true) {
      const calc = await this.get(calculationId);

      if (calc.status === 'completed' || calc.status === 'failed') {
        return calc;
      }

      if (timeout && (Date.now() - startTime) > timeout * 1000) {
        throw new Error(`Calculation ${calculationId} did not complete within ${timeout}s`);
      }

      await this._sleep(pollInterval);
    }
  }

  /**
   * Submit batch of calculations
   * @param {string} projectId - Project ID
   * @param {Array} calculations - Array of calculation data
   * @returns {Promise<Object>} Batch submission result
   */
  async submitBatch(projectId, calculations) {
    const data = {
      project_id: projectId,
      calculations,
    };

    return await this.client.post(`/api/v1/projects/${projectId}/calculations/batch`, data);
  }

  _sleep(ms) {
    return new Promise(resolve => setTimeout(resolve, ms));
  }
}

/**
 * Webhooks API
 */
class WebhooksAPI {
  constructor(client) {
    this.client = client;
  }

  /**
   * Subscribe to webhook events
   * @param {Object} data - Subscription data
   * @returns {Promise<Object>} Subscription
   */
  async subscribe(data) {
    const payload = {
      url: data.url,
      events: data.events,
      metadata: data.metadata || {},
    };

    return await this.client.post('/api/v1/webhooks/subscribe', payload);
  }

  /**
   * List webhook subscriptions
   * @returns {Promise<Array>} List of subscriptions
   */
  async list() {
    return await this.client.get('/api/v1/webhooks');
  }

  /**
   * Delete webhook subscription
   * @param {string} webhookId - Webhook ID
   * @returns {Promise<void>}
   */
  async delete(webhookId) {
    await this.client.delete(`/api/v1/webhooks/${webhookId}`);
  }

  /**
   * Verify webhook signature
   * @param {string} payload - Raw request body
   * @param {string} signature - Signature from X-Webhook-Signature header
   * @param {string} secret - Webhook secret
   * @returns {boolean} True if signature is valid
   */
  static verifySignature(payload, signature, secret) {
    const expected = crypto
      .createHmac('sha256', secret)
      .update(payload)
      .digest('hex');
    
    const expectedSig = `sha256=${expected}`;
    
    try {
      return crypto.timingSafeEqual(
        Buffer.from(expectedSig),
        Buffer.from(signature)
      );
    } catch {
      return false;
    }
  }
}

/**
 * Main SDK Client
 */
class Client {
  /**
   * Create a new API client
   * @param {Object} options - Client options
   * @param {string} options.apiKey - API key
   * @param {string} options.baseURL - API base URL
   * @param {number} options.timeout - Request timeout in ms
   * @param {number} options.maxRetries - Maximum retries
   */
  constructor(options = {}) {
    this._client = new HTTPClient(options);

    // API resources
    this.projects = new ProjectsAPI(this._client);
    this.calculations = new CalculationsAPI(this._client);
    this.webhooks = new WebhooksAPI(this._client);
  }

  /**
   * Check API health
   * @returns {Promise<Object>} Health status
   */
  async health() {
    return await this._client.get('/health');
  }

  /**
   * Get usage statistics
   * @returns {Promise<Object>} Usage stats
   */
  async usage() {
    return await this._client.get('/api/v1/usage');
  }
}

// Export for different module systems
module.exports = {
  Client,
  DFTLAMMPSError,
  AuthenticationError,
  RateLimitError,
  NotFoundError,
  ValidationError,
  WebhooksAPI,
};

// Default export
module.exports.default = Client;
