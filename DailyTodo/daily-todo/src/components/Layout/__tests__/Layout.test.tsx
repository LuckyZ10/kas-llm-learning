import { describe, it, expect } from 'vitest';
import '@testing-library/jest-dom';
import { render, screen } from '@testing-library/react';
import { Layout } from '../Layout';

describe('Layout', () => {
  it('should render header, date selectors, and main content', () => {
    render(<Layout>Test Content</Layout>);
    
    expect(screen.getByTestId('date-selector')).toBeInTheDocument();
    expect(screen.getByTestId('main-content')).toBeInTheDocument();
  });
});