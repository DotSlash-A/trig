import { mutation, query } from "./_generated/server";
import { v } from "convex/values";
import { getAuthUserId } from "@convex-dev/auth/server";

// Enhanced tokenizer to handle more complex expressions
function tokenize(expr: string): string[] {
  const pattern = /(\d+\.?\d*|[\+\-\*\/\^\(\)]|sin|cos|tan|cot|sec|cosec|log|ln|e|pi|x|lim|infinity|∞)/g;
  return (expr.match(pattern) || []).map(token => 
    token === "infinity" || token === "∞" ? "Infinity" : token
  );
}

// Constants
const PI = Math.PI;
const E = Math.E;

function isNumber(token: string): boolean {
  return !isNaN(Number(token)) || token === "Infinity" || token === "-Infinity";
}

function isOperator(token: string): boolean {
  return ['+', '-', '*', '/', '^'].includes(token);
}

function getPrecedence(op: string): number {
  switch (op) {
    case '+':
    case '-':
      return 1;
    case '*':
    case '/':
      return 2;
    case '^':
      return 3;
    default:
      return 0;
  }
}

// Enhanced expression evaluator with special cases
function evaluateExpression(tokens: string[], x: number): number {
  const values: number[] = [];
  const ops: string[] = [];

  function applyOp(): void {
    const op = ops.pop()!;
    const b = values.pop()!;
    const a = values.pop()!;
    
    switch (op) {
      case '+':
        values.push(a + b);
        break;
      case '-':
        values.push(a - b);
        break;
      case '*':
        values.push(a * b);
        break;
      case '/':
        if (b === 0) {
          if (a > 0) values.push(Infinity);
          else if (a < 0) values.push(-Infinity);
          else values.push(NaN);
        } else {
          values.push(a / b);
        }
        break;
      case '^':
        // Handle special cases for exponents
        if (a === 0 && b < 0) values.push(Infinity);
        else if (a === 0 && b === 0) values.push(NaN);
        else if (a === 1) values.push(1);
        else if (b === 0) values.push(1);
        else if (isFinite(a) && isFinite(b)) values.push(Math.pow(a, b));
        else {
          // Handle limits of the form 1^∞
          if (a === 1 && !isFinite(b)) values.push(1);
          // Handle limits of the form 0^0
          else if (a === 0 && b === 0) values.push(NaN);
          else values.push(Math.pow(a, b));
        }
        break;
    }
  }

  for (const token of tokens) {
    if (token === 'x') {
      values.push(x);
    } else if (token === 'pi' || token === 'π') {
      values.push(PI);
    } else if (token === 'e') {
      values.push(E);
    } else if (isNumber(token)) {
      values.push(Number(token));
    } else if (token === '(') {
      ops.push(token);
    } else if (token === ')') {
      while (ops.length && ops[ops.length - 1] !== '(') {
        applyOp();
      }
      ops.pop(); // Remove '('
    } else if (['sin', 'cos', 'tan', 'cot', 'sec', 'cosec', 'log', 'ln'].includes(token)) {
      const val = values.pop()!;
      switch (token) {
        case 'sin':
          values.push(Math.sin(val));
          break;
        case 'cos':
          values.push(Math.cos(val));
          break;
        case 'tan':
          values.push(Math.tan(val));
          break;
        case 'cot':
          values.push(1 / Math.tan(val));
          break;
        case 'sec':
          values.push(1 / Math.cos(val));
          break;
        case 'cosec':
          values.push(1 / Math.sin(val));
          break;
        case 'log':
          values.push(Math.log10(val));
          break;
        case 'ln':
          values.push(Math.log(val));
          break;
      }
    } else if (isOperator(token)) {
      while (
        ops.length &&
        ops[ops.length - 1] !== '(' &&
        getPrecedence(ops[ops.length - 1]) >= getPrecedence(token)
      ) {
        applyOp();
      }
      ops.push(token);
    }
  }

  while (ops.length) {
    applyOp();
  }

  return values[0];
}

// Function to check if expression is of form 1^∞
function isOnePowerInfinity(expr: string, x: number): boolean {
  const parts = expr.split('^');
  if (parts.length !== 2) return false;
  
  try {
    const base = evaluateExpression(tokenize(parts[0]), x);
    const exponent = evaluateExpression(tokenize(parts[1]), x);
    return Math.abs(base - 1) < 1e-10 && !isFinite(exponent);
  } catch {
    return false;
  }
}

// Function to handle special trigonometric limits
function handleTrigLimit(expr: string, x: number): number | null {
  // Handle sin(x)/x as x → 0
  if (expr.match(/sin\(x\)\/x/) && x === 0) {
    return 1;
  }
  
  // Handle (1-cos(x))/x^2 as x → 0
  if (expr.match(/\(1-cos\(x\)\)\/\(x\^2\)/) && x === 0) {
    return 0.5;
  }
  
  // Handle tan(x)/x as x → 0
  if (expr.match(/tan\(x\)\/x/) && x === 0) {
    return 1;
  }

  return null;
}

// Function to handle factorization method
function factorize(expr: string): string | null {
  // Handle (x^2-1)/(x-1) → (x+1)
  if (expr.match(/\(x\^2-1\)\/\(x-1\)/)) {
    return "(x+1)";
  }
  
  // Handle more factorization cases here
  return null;
}

// Function to handle rationalization
function rationalize(expr: string): string | null {
  // Handle sqrt expressions
  if (expr.includes('sqrt')) {
    // Implement rationalization logic
    return null;
  }
  return null;
}

function calculateLimit(expr: string, variable: string, tendingTo: string): { 
  steps: Array<{step: string, explanation: string}>, 
  result: string 
} {
  const steps: Array<{step: string, explanation: string}> = [];
  const tokens = tokenize(expr);
  
  // Handle infinity
  const limit = tendingTo === "infinity" || tendingTo === "∞" ? 
    Infinity : 
    tendingTo === "-infinity" || tendingTo === "-∞" ? 
    -Infinity : 
    Number(tendingTo);

  if (tendingTo !== "infinity" && tendingTo !== "∞" && tendingTo !== "-infinity" && tendingTo !== "-∞" && isNaN(limit)) {
    return {
      steps: [{
        step: "Invalid input",
        explanation: "The limit value must be a number or infinity"
      }],
      result: "Error"
    };
  }

  steps.push({
    step: `Evaluating limit of ${expr} as ${variable} → ${tendingTo}`,
    explanation: "Starting the limit evaluation process"
  });

  // Check for special trigonometric limits
  const trigResult = handleTrigLimit(expr, limit);
  if (trigResult !== null) {
    steps.push({
      step: "Special trigonometric limit detected",
      explanation: "Using standard trigonometric limit formulas"
    });
    return {
      steps,
      result: trigResult.toString()
    };
  }

  // Try factorization
  const factorized = factorize(expr);
  if (factorized) {
    steps.push({
      step: "Factorizing the expression",
      explanation: `Simplified to ${factorized}`
    });
    expr = factorized;
  }

  // Try rationalization
  const rationalized = rationalize(expr);
  if (rationalized) {
    steps.push({
      step: "Rationalizing the expression",
      explanation: `Rationalized to ${rationalized}`
    });
    expr = rationalized;
  }

  // Check for 1^∞ form
  if (isOnePowerInfinity(expr, limit)) {
    steps.push({
      step: "Detected form 1^∞",
      explanation: "Using the standard limit formula for 1^∞"
    });
    return {
      steps,
      result: "1"
    };
  }

  // Try direct substitution
  try {
    const directValue = evaluateExpression(tokens, limit);
    if (!isNaN(directValue) && isFinite(directValue)) {
      steps.push({
        step: `Direct substitution: ${variable} = ${limit}`,
        explanation: "The limit exists and can be found by direct substitution"
      });
      return { 
        steps,
        result: directValue.toString()
      };
    }
  } catch (e) {
    steps.push({
      step: "Direct substitution failed",
      explanation: "Cannot directly substitute the limit value"
    });
  }

  // Check left and right hand limits for finite values
  if (isFinite(limit)) {
    try {
      const epsilon = 0.000001;
      const leftValue = evaluateExpression(tokens, limit - epsilon);
      const rightValue = evaluateExpression(tokens, limit + epsilon);
      
      steps.push({
        step: `Checking left-hand limit: ${leftValue}`,
        explanation: `Evaluated as ${variable} approaches ${limit} from left`
      });
      steps.push({
        step: `Checking right-hand limit: ${rightValue}`,
        explanation: `Evaluated as ${variable} approaches ${limit} from right`
      });

      if (Math.abs(leftValue - rightValue) < epsilon) {
        return {
          steps: [...steps, {
            step: `Left and right limits are equal`,
            explanation: `The limit exists and equals ${leftValue}`
          }],
          result: leftValue.toString()
        };
      } else {
        return {
          steps: [...steps, {
            step: "Left and right limits are not equal",
            explanation: "The limit does not exist"
          }],
          result: "DNE"
        };
      }
    } catch (e) {
      // Continue with other methods if this fails
    }
  }

  // Handle infinity limits
  if (!isFinite(limit)) {
    try {
      const value = evaluateExpression(tokens, limit > 0 ? 1e10 : -1e10);
      if (isFinite(value)) {
        return {
          steps: [...steps, {
            step: "Evaluating at a very large value",
            explanation: `The limit approaches ${value}`
          }],
          result: value.toString()
        };
      } else {
        return {
          steps: [...steps, {
            step: "Limit tends to infinity",
            explanation: value > 0 ? "Positive infinity" : "Negative infinity"
          }],
          result: value > 0 ? "∞" : "-∞"
        };
      }
    } catch (e) {
      // Continue with other methods if this fails
    }
  }

  return {
    steps: [...steps, {
      step: "No conclusive result",
      explanation: "Could not determine the limit using available methods"
    }],
    result: "Undefined"
  };
}

export const calculate = mutation({
  args: {
    expression: v.string(),
    variable: v.string(),
    tendingTo: v.string(),
  },
  handler: async (ctx, args) => {
    const userId = await getAuthUserId(ctx);
    if (!userId) {
      throw new Error("Not authenticated");
    }

    const result = calculateLimit(args.expression, args.variable, args.tendingTo);
    
    await ctx.db.insert("calculations", {
      expression: args.expression,
      variable: args.variable,
      tendingTo: args.tendingTo,
      steps: result.steps,
      result: result.result,
      userId
    });

    return result;
  },
});

export const getCalculations = query({
  args: {},
  handler: async (ctx) => {
    const userId = await getAuthUserId(ctx);
    if (!userId) {
      return [];
    }

    return await ctx.db
      .query("calculations")
      .withIndex("by_user", (q) => q.eq("userId", userId))
      .order("desc")
      .collect();
  },
});

aboveis calculator.ts

write this code in fast api, so that nom atter what expression i give, my limits calcualtor shoudl respond witht he steps by step solution.
a noob has written this logic, you beign expereinced, can do additions to the logic

the goal:
you should be able to take nay input expression, give user the option to input lmit tending to what, then numerator, denominator. shoudl include all types of buttons, trignometric, algebric, etc. so the user shoudl be ablt ot put in any function.
next when the user inputs and clicks "calculate" button, you should be able to calculate limits (use class 12/bachelors level) knowlwdge to solve the sum.
shoudl show sterp by step calculation and solution for that sum.
In the backedn I want you to write all the calcuation lgic by yourself, like parsing the input, finding teh limits, knowing whihc method to use to slve that question, then solving and showig soluiotn. Ideally I do not want you to use any library like sympy but instead write everything like solving logic by yourself.
also modify the backedn such that it can pretty musch slve any type of expression, factorization method, ratoinalization methos, evaluaiton at infinity, trgi limits when variable tends to non zero, trigonomtery limits by factorization,
exponential, evaluation of limits of the form 1 power infinity.

make sure you write al the logic for solving these types by yourself, correctly, the ans should be corect

dot use sympy or any external ibrary, I want you to write everything from scratch. so athte end I will five quarion as a json, app shoudl give the step by step soltuon as response


also explaint he logic, this sperson has wriiten something, but I dont understand how, so if I hve to wrtie again by myself, I nee dto understand the logic.
if you are making nay changes and improvign the logic tell what changes you have done
for now this is just either checking left and rihtg hand limits it hing, but make sure you improve to use different types of methods that I mentione dot solve then sums.
make this an advanced type of calculator.

https://chef.show/4bde38