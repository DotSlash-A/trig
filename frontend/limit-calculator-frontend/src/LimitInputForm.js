// src/LimitInputForm.js
import React from 'react';

function LimitInputForm({
  expression,
  variable,
  tendingTo,
  onExpressionChange,
  onVariableChange,
  onTendingToChange,
  onSubmit,
  isLoading,
}) {
  const handleFormSubmit = (e) => {
    e.preventDefault(); // Prevent default form submission
    onSubmit();
  };

  return (
    <form className="limit-input-form" onSubmit={handleFormSubmit}>
      <label htmlFor="expression">lim</label>
      <input
        type="text"
        id="expression"
        className="expression-input"
        value={expression}
        onChange={onExpressionChange}
        placeholder="e.g., sin(x)/x or (x^2-1)/(x-1)"
        required
        disabled={isLoading}
      />
      <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', lineHeight: '1' }}>
        <input
          type="text"
          id="variable"
          className="variable-input"
          value={variable}
          onChange={onVariableChange}
          required
          disabled={isLoading}
          maxLength={5} // Limit variable length
        />
        <span>&rarr;</span> {/* Arrow symbol */}
        <input
          type="text"
          id="tendingTo"
          className="tending-to-input"
          value={tendingTo}
          onChange={onTendingToChange}
          placeholder="0" // Placeholder for limit point
          required
          disabled={isLoading}
        />
      </div>
      <button type="submit" disabled={isLoading}>
        {isLoading ? 'Calculating...' : 'Go'}
      </button>
    </form>
  );
}

export default LimitInputForm;