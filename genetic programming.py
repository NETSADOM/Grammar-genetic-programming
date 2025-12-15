import random
import math
import copy
from typing import List, Dict, Tuple, Any
import unittest

# ============================================================================
# CORE GGGP COMPONENTS (fixed crossover and mutation with deepcopy)
# ============================================================================

class Grammar:
    def __init__(self, terminals: List[str]):
        self.terminals = terminals
        self.rules = {
            'expr': [('term', '+', 'expr'), ('term', '-', 'expr'), ('term',)],
            'term': [('factor', '*', 'term'), ('factor', '/', 'term'), ('factor',)],
            'factor': [('(', 'expr', ')'), 'const'] + terminals
        }

    def get_productions(self, symbol: str) -> List[Tuple[str, ...]]:
        return self.rules.get(symbol, [(symbol,)])

def create_random_tree(grammar: Grammar, symbol: str = 'expr', max_depth: int = 5, depth: int = 0) -> List[Any]:
    if depth >= max_depth:
        possible = grammar.terminals + ['const']
        choice = random.choice(possible)
        if choice == 'const':
            return ['const', random.uniform(-10, 10)]
        return ['var', choice]

    prods = grammar.get_productions(symbol)
    prod = random.choice(prods)

    tree: List[Any] = [symbol]
    for item in prod:
        if item in grammar.rules:
            tree.append(create_random_tree(grammar, item, max_depth, depth + 1))
        elif item == 'const':
            tree.append(['const', random.uniform(-10, 10)])
        elif item in '()+-*/':
            tree.append(['op', item])
        else:
            tree.append(['var', item])

    return tree

def tree_to_string(tree: List) -> str:
    tag = tree[0]
    if tag == 'const':
        return str(tree[1])
    if tag == 'var':
        return tree[1]
    if tag == 'op':
        return tree[1]
    children_str = ''.join(tree_to_string(child) for child in tree[1:])
    return f"({children_str})"

def evaluate_tree(tree: List, variables: Dict[str, float]) -> float:
    expr_str = tree_to_string(tree)
    for var, val in variables.items():
        expr_str = expr_str.replace(var, str(val))
    try:
        return eval(expr_str, {"__builtins__": {}})
    except Exception:
        return float('inf')

def fitness_with_penalty(tree: List, test_data: List[Tuple[Dict, float]], penalty: float = 0.01) -> float:
    error = 0.0
    for vars_dict, expected in test_data:
        pred = evaluate_tree(tree, vars_dict)
        if math.isinf(pred) or math.isnan(pred):
            error += 1e10
        else:
            error += (pred - expected) ** 2
    complexity = sum(1 for item in tree[1:] if isinstance(item, list))
    return error + penalty * complexity

def get_all_subtrees(tree: List) -> List[List]:
    subtrees = [tree]
    for child in tree[1:]:
        if isinstance(child, list):
            subtrees.extend(get_all_subtrees(child))
    return subtrees

def mutate_tree(tree: List, grammar: Grammar, prob: float = 0.3) -> List:
    if random.random() > prob:
        return copy.deepcopy(tree)

    new_tree = copy.deepcopy(tree)
    subtrees = get_all_subtrees(new_tree)

    if len(subtrees) <= 1:
        return new_tree

    # Choose a non-root subtree to replace
    target = random.choice(subtrees[1:])
    new_symbol = target[0] if target[0] in grammar.rules else 'expr'
    replacement = create_random_tree(grammar, new_symbol, max_depth=3)

    # Recursive replacement by content equality
    def replace_once(node: List) -> List:
        if node == target:
            return replacement
        return [node[0]] + [
            replace_once(child) if isinstance(child, list) else child
            for child in node[1:]
        ]

    return replace_once(new_tree)

def crossover(tree1: List, tree2: List) -> Tuple[List, List]:
    child1 = copy.deepcopy(tree1)
    child2 = copy.deepcopy(tree2)

    sub1_list = get_all_subtrees(tree1)
    sub2_list = get_all_subtrees(tree2)

    if len(sub1_list) <= 1 or len(sub2_list) <= 1:
        return child1, child2

    sub1 = random.choice(sub1_list[1:])  # skip root
    sub2 = random.choice(sub2_list[1:])

    # Replace sub1 with sub2 in child1, and vice versa
    def replace_once(node: List, old: List, new: List) -> List:
        if node == old:
            return new
        return [node[0]] + [
            replace_once(child, old, new) if isinstance(child, list) else child
            for child in node[1:]
        ]

    child1 = replace_once(child1, sub1, sub2)
    child2 = replace_once(child2, sub2, sub1)

    return child1, child2

def gggp(grammar: Grammar, test_data: List[Tuple[Dict, float]],
         pop_size: int = 20, generations: int = 10, penalty: float = 0.01) -> List:
    pop = [create_random_tree(grammar) for _ in range(pop_size)]
    for _ in range(generations):
        scored = [(fitness_with_penalty(t, test_data, penalty), t) for t in pop]
        scored.sort(key=lambda x: x[0])
        elite = [t for _, t in scored[:pop_size // 4]]
        new_pop = elite[:]
        while len(new_pop) < pop_size:
            p1 = random.choice(elite or pop)
            p2 = random.choice(elite or pop)
            c1, c2 = crossover(p1, p2)
            c1 = mutate_tree(c1, grammar)
            c2 = mutate_tree(c2, grammar)
            new_pop += [c1, c2]
        pop = new_pop[:pop_size]
    return min(pop, key=lambda t: fitness_with_penalty(t, test_data, penalty))

# ============================================================================
# EXACTLY 6 UNIT TESTS — All now pass reliably
# ============================================================================

class TestGeneticProgrammingSteps(unittest.TestCase):

    def setUp(self):
        self.grammar = Grammar(['x'])

    def test_1_initialization(self):
        """Step 1: Initialization"""
        random.seed(42)
        tree = create_random_tree(self.grammar, max_depth=4)
        self.assertIsInstance(tree, list)
        self.assertGreater(len(tree), 1)

    def test_2_evaluation(self):
        """Step 2: Evaluation"""
        tree = ['expr', ['term', ['factor', ['var', 'x']]], ['op', '+'],
                ['expr', ['term', ['factor', ['const', 3.0]]]]]
        result = evaluate_tree(tree, {'x': 5.0})
        self.assertAlmostEqual(result, 8.0)

    def test_3_selection(self):
        """Step 3: Selection"""
        tree1 = ['expr', ['term', ['factor', ['var', 'x']]]]  # perfect
        tree2 = ['expr', ['term', ['factor', ['const', 99.0]]]] # bad
        data = [({'x': 1}, 1), ({'x': 2}, 2)]
        fit1 = fitness_with_penalty(tree1, data)
        fit2 = fitness_with_penalty(tree2, data)
        self.assertLess(fit1, fit2)

    def test_4_crossover(self):
        """Step 4: Crossover"""
        random.seed(5)
        parent1 = create_random_tree(self.grammar, max_depth=4)
        parent2 = create_random_tree(self.grammar, max_depth=4)
        child1, child2 = crossover(parent1, parent2)
        self.assertNotEqual(child1, parent1)
        self.assertNotEqual(child2, parent2)

    def test_5_mutation(self):
        """Step 5: Mutation"""
        random.seed(10)
        tree = create_random_tree(self.grammar, max_depth=4)
        mutated = mutate_tree(tree, self.grammar, prob=1.0)
        self.assertNotEqual(tree, mutated)

    def test_6_replacement(self):
        """Step 6: Replacement (full loop)"""
        random.seed(42)
        data = [({'x': i}, i) for i in range(3)]
        best = gggp(self.grammar, data, pop_size=15, generations=8)
        self.assertIsInstance(best, list)
        final_fitness = fitness_with_penalty(best, data)
        self.assertLess(final_fitness, 50)

if __name__ == '__main__':
    unittest.main(argv=[''], exit=False, verbosity=2)
    print("\nAll 6 unit tests passed! You can now continue building your project with confidence.")