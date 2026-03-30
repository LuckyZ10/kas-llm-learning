// Package dftlammps provides a Go client for the DFT+LAMMPS API Platform.
//
// Installation:
//   go get github.com/dft-lammps/go-client
//
// Quick Start:
//   client := dftlammps.NewClient("your-api-key")
//   project, err := client.Projects.Create(ctx, &CreateProjectRequest{
//       Name: "My Project",
//   })
//
package dftlammps

import (
	"bytes"
	"context"
	"crypto/hmac"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"os"
	"strconv"
	"time"
)

const (
	defaultBaseURL    = "https://api.dft-lammps.org"
	defaultTimeout    = 30 * time.Second
	defaultMaxRetries = 3
	version           = "1.0.0"
)

// Client is the main API client
type Client struct {
	HTTPClient *http.Client
	BaseURL    string
	APIKey     string

	Projects     *ProjectsService
	Calculations *CalculationsService
	Webhooks     *WebhooksService
}

// ClientOption allows customization of the client
type ClientOption func(*Client)

// WithHTTPClient sets a custom HTTP client
func WithHTTPClient(httpClient *http.Client) ClientOption {
	return func(c *Client) {
		c.HTTPClient = httpClient
	}
}

// WithBaseURL sets a custom base URL
func WithBaseURL(baseURL string) ClientOption {
	return func(c *Client) {
		c.BaseURL = baseURL
	}
}

// NewClient creates a new API client
func NewClient(apiKey string, opts ...ClientOption) (*Client, error) {
	if apiKey == "" {
		apiKey = os.Getenv("DFT_LAMMPS_API_KEY")
	}
	if apiKey == "" {
		return nil, fmt.Errorf("API key required: set DFT_LAMMPS_API_KEY or pass to constructor")
	}

	client := &Client{
		HTTPClient: &http.Client{Timeout: defaultTimeout},
		BaseURL:    defaultBaseURL,
		APIKey:     apiKey,
	}

	for _, opt := range opts {
		opt(client)
	}

	// Initialize services
	client.Projects = &ProjectsService{client: client}
	client.Calculations = &CalculationsService{client: client}
	client.Webhooks = &WebhooksService{client: client}

	return client, nil
}

// doRequest makes an HTTP request with retries
func (c *Client) doRequest(ctx context.Context, method, path string, body interface{}, params url.Values) (*http.Response, error) {
	var bodyReader io.Reader
	if body != nil {
		jsonBody, err := json.Marshal(body)
		if err != nil {
			return nil, err
		}
		bodyReader = bytes.NewReader(jsonBody)
	}

	// Build URL
	u, err := url.Parse(c.BaseURL + path)
	if err != nil {
		return nil, err
	}
	if params != nil {
		u.RawQuery = params.Encode()
	}

	// Create request
	req, err := http.NewRequestWithContext(ctx, method, u.String(), bodyReader)
	if err != nil {
		return nil, err
	}

	// Set headers
	req.Header.Set("Authorization", "Bearer "+c.APIKey)
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("User-Agent", "dft-lammps-go/"+version)

	// Execute with retries
	var resp *http.Response
	for attempt := 0; attempt <= defaultMaxRetries; attempt++ {
		resp, err = c.HTTPClient.Do(req)
		if err == nil && resp.StatusCode < 500 && resp.StatusCode != 429 {
			break
		}

		if attempt < defaultMaxRetries {
			time.Sleep(time.Duration(attempt+1) * time.Second)
		}
	}

	return resp, err
}

// decodeResponse decodes JSON response
func decodeResponse(resp *http.Response, v interface{}) error {
	defer resp.Body.Close()

	if resp.StatusCode >= 400 {
		body, _ := io.ReadAll(resp.Body)
		return &APIError{
			StatusCode: resp.StatusCode,
			Message:    string(body),
		}
	}

	if v != nil {
		return json.NewDecoder(resp.Body).Decode(v)
	}

	return nil
}

// APIError represents an API error
type APIError struct {
	StatusCode int
	Message    string
}

func (e *APIError) Error() string {
	return fmt.Sprintf("API error %d: %s", e.StatusCode, e.Message)
}

// IsNotFound returns true for 404 errors
func (e *APIError) IsNotFound() bool {
	return e.StatusCode == 404
}

// IsRateLimit returns true for 429 errors
func (e *APIError) IsRateLimit() bool {
	return e.StatusCode == 429
}

// IsAuthError returns true for 401/403 errors
func (e *APIError) IsAuthError() bool {
	return e.StatusCode == 401 || e.StatusCode == 403
}

// ==================== Projects API ====================

// ProjectsService handles project operations
type ProjectsService struct {
	client *Client
}

// Project represents a project
type Project struct {
	ID                    string                 `json:"id"`
	Name                  string                 `json:"name"`
	Description           string                 `json:"description"`
	ProjectType           string                 `json:"project_type"`
	Status                string                 `json:"status"`
	TargetProperties      map[string]interface{} `json:"target_properties"`
	MaterialSystem        string                 `json:"material_system"`
	Tags                  []string               `json:"tags"`
	TotalStructures       int                    `json:"total_structures"`
	CompletedCalculations int                    `json:"completed_calculations"`
	FailedCalculations    int                    `json:"failed_calculations"`
	CreatedAt             time.Time              `json:"created_at"`
	UpdatedAt             *time.Time             `json:"updated_at"`
	OwnerID               string                 `json:"owner_id"`
}

// ProjectList represents a paginated list of projects
type ProjectList struct {
	Items    []Project `json:"items"`
	Total    int       `json:"total"`
	Page     int       `json:"page"`
	PageSize int       `json:"page_size"`
	HasNext  bool      `json:"has_next"`
	HasPrev  bool      `json:"has_prev"`
}

// ListProjectsRequest represents list parameters
type ListProjectsRequest struct {
	Status      string
	ProjectType string
	Search      string
	Page        int
	PageSize    int
}

// List retrieves projects with pagination
func (s *ProjectsService) List(ctx context.Context, req *ListProjectsRequest) (*ProjectList, error) {
	params := url.Values{}
	if req.Page > 0 {
		params.Set("page", strconv.Itoa(req.Page))
	}
	if req.PageSize > 0 {
		params.Set("page_size", strconv.Itoa(req.PageSize))
	}
	if req.Status != "" {
		params.Set("status", req.Status)
	}
	if req.ProjectType != "" {
		params.Set("project_type", req.ProjectType)
	}
	if req.Search != "" {
		params.Set("search", req.Search)
	}

	resp, err := s.client.doRequest(ctx, "GET", "/api/v1/projects", nil, params)
	if err != nil {
		return nil, err
	}

	var result ProjectList
	if err := decodeResponse(resp, &result); err != nil {
		return nil, err
	}

	return &result, nil
}

// CreateProjectRequest represents project creation parameters
type CreateProjectRequest struct {
	Name             string                 `json:"name"`
	Description      string                 `json:"description,omitempty"`
	ProjectType      string                 `json:"project_type,omitempty"`
	TargetProperties map[string]interface{} `json:"target_properties,omitempty"`
	MaterialSystem   string                 `json:"material_system,omitempty"`
	Tags             []string               `json:"tags,omitempty"`
}

// Create creates a new project
func (s *ProjectsService) Create(ctx context.Context, req *CreateProjectRequest) (*Project, error) {
	resp, err := s.client.doRequest(ctx, "POST", "/api/v1/projects", req, nil)
	if err != nil {
		return nil, err
	}

	var project Project
	if err := decodeResponse(resp, &project); err != nil {
		return nil, err
	}

	return &project, nil
}

// Get retrieves a project by ID
func (s *ProjectsService) Get(ctx context.Context, projectID string) (*Project, error) {
	resp, err := s.client.doRequest(ctx, "GET", "/api/v1/projects/"+projectID, nil, nil)
	if err != nil {
		return nil, err
	}

	var project Project
	if err := decodeResponse(resp, &project); err != nil {
		return nil, err
	}

	return &project, nil
}

// UpdateProjectRequest represents project update parameters
type UpdateProjectRequest struct {
	Name             string                 `json:"name,omitempty"`
	Description      string                 `json:"description,omitempty"`
	Status           string                 `json:"status,omitempty"`
	TargetProperties map[string]interface{} `json:"target_properties,omitempty"`
	Tags             []string               `json:"tags,omitempty"`
}

// Update updates a project
func (s *ProjectsService) Update(ctx context.Context, projectID string, req *UpdateProjectRequest) (*Project, error) {
	resp, err := s.client.doRequest(ctx, "PATCH", "/api/v1/projects/"+projectID, req, nil)
	if err != nil {
		return nil, err
	}

	var project Project
	if err := decodeResponse(resp, &project); err != nil {
		return nil, err
	}

	return &project, nil
}

// Delete deletes a project
func (s *ProjectsService) Delete(ctx context.Context, projectID string) error {
	resp, err := s.client.doRequest(ctx, "DELETE", "/api/v1/projects/"+projectID, nil, nil)
	if err != nil {
		return err
	}

	return decodeResponse(resp, nil)
}

// ==================== Calculations API ====================

// CalculationsService handles calculation operations
type CalculationsService struct {
	client *Client
}

// Calculation represents a calculation
type Calculation struct {
	ID               string                 `json:"id"`
	ProjectID        string                 `json:"project_id"`
	CalculationType  string                 `json:"calculation_type"`
	Status           string                 `json:"status"`
	Structure        map[string]interface{} `json:"structure"`
	Parameters       map[string]interface{} `json:"parameters"`
	Priority         int                    `json:"priority"`
	Results          map[string]interface{} `json:"results"`
	ErrorMessage     string                 `json:"error_message"`
	CreatedAt        time.Time              `json:"created_at"`
	StartedAt        *time.Time             `json:"started_at"`
	CompletedAt      *time.Time             `json:"completed_at"`
}

// SubmitCalculationRequest represents calculation submission
type SubmitCalculationRequest struct {
	Structure       map[string]interface{} `json:"structure"`
	CalculationType string                 `json:"calculation_type"`
	Parameters      map[string]interface{} `json:"parameters,omitempty"`
	Priority        int                    `json:"priority,omitempty"`
}

// Submit creates a new calculation
func (s *CalculationsService) Submit(ctx context.Context, projectID string, req *SubmitCalculationRequest) (*Calculation, error) {
	resp, err := s.client.doRequest(ctx, "POST", "/api/v1/projects/"+projectID+"/calculations", req, nil)
	if err != nil {
		return nil, err
	}

	var calc Calculation
	if err := decodeResponse(resp, &calc); err != nil {
		return nil, err
	}

	return &calc, nil
}

// Get retrieves a calculation by ID
func (s *CalculationsService) Get(ctx context.Context, calculationID string) (*Calculation, error) {
	resp, err := s.client.doRequest(ctx, "GET", "/api/v1/calculations/"+calculationID, nil, nil)
	if err != nil {
		return nil, err
	}

	var calc Calculation
	if err := decodeResponse(resp, &calc); err != nil {
		return nil, err
	}

	return &calc, nil
}

// Wait polls until calculation completes
func (s *CalculationsService) Wait(ctx context.Context, calculationID string, pollInterval time.Duration) (*Calculation, error) {
	if pollInterval == 0 {
		pollInterval = 5 * time.Second
	}

	ticker := time.NewTicker(pollInterval)
	defer ticker.Stop()

	for {
		select {
		case <-ctx.Done():
			return nil, ctx.Err()
		case <-ticker.C:
			calc, err := s.Get(ctx, calculationID)
			if err != nil {
				return nil, err
			}

			if calc.Status == "completed" || calc.Status == "failed" {
				return calc, nil
			}
		}
	}
}

// ==================== Webhooks API ====================

// WebhooksService handles webhook operations
type WebhooksService struct {
	client *Client
}

// WebhookSubscription represents a webhook subscription
type WebhookSubscription struct {
	WebhookID string            `json:"webhook_id"`
	URL       string            `json:"url"`
	Events    []string          `json:"events"`
	Secret    string            `json:"secret"`
	Status    string            `json:"status"`
	CreatedAt string            `json:"created_at"`
	Metadata  map[string]string `json:"metadata,omitempty"`
}

// Subscribe creates a new webhook subscription
func (s *WebhooksService) Subscribe(ctx context.Context, url string, events []string, metadata map[string]string) (*WebhookSubscription, error) {
	req := map[string]interface{}{
		"url":      url,
		"events":   events,
		"metadata": metadata,
	}

	resp, err := s.client.doRequest(ctx, "POST", "/api/v1/webhooks/subscribe", req, nil)
	if err != nil {
		return nil, err
	}

	var sub WebhookSubscription
	if err := decodeResponse(resp, &sub); err != nil {
		return nil, err
	}

	return &sub, nil
}

// List retrieves all webhook subscriptions
func (s *WebhooksService) List(ctx context.Context) ([]WebhookSubscription, error) {
	resp, err := s.client.doRequest(ctx, "GET", "/api/v1/webhooks", nil, nil)
	if err != nil {
		return nil, err
	}

	var subs []WebhookSubscription
	if err := decodeResponse(resp, &subs); err != nil {
		return nil, err
	}

	return subs, nil
}

// Delete removes a webhook subscription
func (s *WebhooksService) Delete(ctx context.Context, webhookID string) error {
	resp, err := s.client.doRequest(ctx, "DELETE", "/api/v1/webhooks/"+webhookID, nil, nil)
	if err != nil {
		return err
	}

	return decodeResponse(resp, nil)
}

// VerifySignature verifies a webhook signature
func VerifySignature(payload []byte, signature, secret string) bool {
	mac := hmac.New(sha256.New, []byte(secret))
	mac.Write(payload)
	expected := "sha256=" + hex.EncodeToString(mac.Sum(nil))

	return hmac.Equal([]byte(expected), []byte(signature))
}

// ==================== Utility Functions ====================

// Health checks API health
func (c *Client) Health(ctx context.Context) (map[string]interface{}, error) {
	resp, err := c.doRequest(ctx, "GET", "/health", nil, nil)
	if err != nil {
		return nil, err
	}

	var result map[string]interface{}
	if err := decodeResponse(resp, &result); err != nil {
		return nil, err
	}

	return result, nil
}

// Usage retrieves usage statistics
func (c *Client) Usage(ctx context.Context) (map[string]interface{}, error) {
	resp, err := c.doRequest(ctx, "GET", "/api/v1/usage", nil, nil)
	if err != nil {
		return nil, err
	}

	var result map[string]interface{}
	if err := decodeResponse(resp, &result); err != nil {
		return nil, err
	}

	return result, nil
}
