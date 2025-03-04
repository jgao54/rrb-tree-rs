#![allow(dead_code)]
#![allow(unused_variables)]

use std::cmp;

// max element size at each slot
const M: usize = 32;
// Extra search step allowed
const E_MAX: usize = 2;

trait RrbElement: Clone + PartialEq + std::fmt::Debug {}

// blanket implementation for all types that meet the requirement
impl<T> RrbElement for T where T: Clone + PartialEq + std::fmt::Debug {}

fn bit_width(m: usize) -> usize {
    (m as f64).log2() as usize
}

#[derive(Debug, Clone)]
struct Rrb<T: RrbElement> {
    count: usize,
    root: Node<T>,    
}

#[derive(Debug, Clone)]
enum Node<T: RrbElement> {
    Branch(Branch<T>),
    Leaf(Leaf<T>),
}

#[derive(Debug, Clone)]
enum NodeOrValue<T: RrbElement> {
    Node(Node<T>),
    Value(T),
}

#[derive(Debug, Clone)]
struct Branch<T: RrbElement> {
    height: usize, // >= 1
    sizes: Vec<usize>,
    items: Vec<Node<T>>,
}

#[derive(Debug, Clone)]
struct Leaf<T: RrbElement> {
    items: Vec<T>,
    height: usize, // always == 0
}

impl<T: RrbElement> Node<T> {
    fn height(&self) -> usize {
        match self {
            Node::Branch(branch) => branch.height,
            Node::Leaf(_) => 0,
        }
    }

    fn length(&self) -> usize {
        match self {
            Node::Branch(branch) => branch.items.len(),
            Node::Leaf(leaf) => leaf.items.len(),
        }
    }

    fn is_branch(&self) -> bool {
        match self {
            Node::Branch(_) => true,
            Node::Leaf(_) => false,
        }
    }

    fn is_leaf(&self) -> bool {
        match self {
            Node::Branch(_) => false,
            Node::Leaf(_) => true,
        }
    }
}

fn leaf<T: RrbElement>(items: Vec<T>) -> Leaf<T> {
    Leaf {
        height: 0, 
        items,
    }
}

fn branch<T: RrbElement>(height: usize, items: Vec<Node<T>>) -> Branch<T> {
    let sizes = calc_sizes(&items);  // calculate sizes for the branch
    Branch {
        height,
        items,
        sizes,
    }
}

fn calc_sizes<T: RrbElement>(items: &Vec<Node<T>>) -> Vec<usize> {
    let mut prev = 0;
    items.iter().map(| i | {
        prev += size_of(i);
        prev
    }).collect()
}

fn size_of<T: RrbElement>(tree: &Node<T>) -> usize {
    match tree {
        Node::Leaf(_) => tree.length(),
        Node::Branch(b) => {
            b.sizes[b.sizes.len()-1]
        }
    }
}

fn init<T: RrbElement>()-> Rrb<T> {
    Rrb {
        count: 0,
        root: Node::Leaf(leaf(vec![])),
    }
}

fn from_array<T: RrbElement>(items: Vec<T>) -> Rrb<T> {
    let mut rrb = init();
    for item in items {
        rrb = append(&rrb, item)
    }
    rrb
}

fn append<T: RrbElement>(xs: &Rrb<T>, x: T) -> Rrb<T> {
    let appended = append_node(&xs.root, x.clone());
    match appended {
        Some(n) => Rrb { count: xs.count + 1, root: n },
        None => {
            let grown = Node::Branch(branch(xs.root.height() + 1, [xs.root.clone()].to_vec()));
            let appended = append_node(&grown, x).unwrap(); // can't fail
            Rrb { count: xs.count + 1, root: appended }
        }
    }
}

// Append `x` to `xs`. Returns null if there is no space.
fn append_node<T: RrbElement>(xs: &Node<T>, x: T) -> Option<Node<T>> {
    match xs {
        Node::Leaf(l) => {
            if l.items.len() == M {
                None
            } else {
                let mut new_items = Vec::with_capacity(l.items.len() + 1);
                new_items.extend_from_slice(&l.items);
                new_items.push(x);
                Some(Node::Leaf(leaf(new_items)))
            }
        }
        Node::Branch(b) => {
            // recursively look for the last node to append to
            let updated: Option<Node<T>> = append_node(b.items.last().unwrap(), x.clone());
            if !updated.is_none() {
                let mut items = b.items[0..b.items.len()-1].to_vec();
                items.push(updated.unwrap());
                Some(Node::Branch(branch(xs.height(), items)))
            } else if b.items.len() < M {
                let mut items = b.items.clone();
                items.push(tree_of_height(xs.height() - 1,  x));
                Some(Node::Branch(branch(xs.height(), items)))
            } else {
                None
            }
        }
    }
}

// Create a tree of `height` with a single element `x`
fn tree_of_height<T: RrbElement>(height: usize, x: T) -> Node<T> {
    if height  == 0 {
        Node::Leaf(leaf([x].to_vec()))
    } else {
        Node::Branch(branch(height, [tree_of_height(height - 1, x)].to_vec()))
    }
}

// Find the element at position `idx`, or None if the index is out of bounds.
fn get<T: RrbElement>(rrb: &Rrb<T>, idx: usize) -> Option<T> {
    if idx >= rrb.count {
        None
    } else {
        Some(find_element(&rrb.root, idx))
    }
}

// Return the idx-th element in the node
fn find_element<T: RrbElement>(node: &Node<T>, idx: usize) -> T {
    match node {
        Node::Branch(branch) => {
            // find the slot containing our element
            let slot = find_slot(idx, branch.height, &branch.sizes);
            // find the number of elements in the preceding slots
            let prev_size = if slot == 0 { 0 } else { branch.sizes[slot - 1] };
            // calculate the index within our slot
            let next_idx = idx - prev_size;
            // recurse
            find_element(&branch.items[slot], next_idx) 
        }
        Node::Leaf(leaf) => {
            // fallback to array indexing for leaf nodes
            leaf.items[idx].clone()
        }
    }
}

// Find the slot at height h given an index
fn find_slot(idx: usize, height: usize, sizes: &Vec<usize>) -> usize {
    // find starting slot by radix indexing
    let w = bit_width(M);
    let mut slot = idx >> (w * height);
    // skip slots until we reach the first with a cumulative size greater
    // than our index - this is where our element will be
    while sizes[slot]  <= idx {
        slot += 1;
    }
    slot
}

// Concat two RRB trees into a single balanced RRB tree
fn concat<T: RrbElement>(left: Rrb<T>, right: Rrb<T>) -> Rrb<T> {
    // create a single, balanced node containing all items from left and right
    let merged = concat_nodes(&left.root, &right.root, true);
    // there may be a redundant extra level so we chop it off if necessary
    let tree = if merged.items.len() == 1  { merged.items[0].clone() } else { Node::Branch(merged) };
    return Rrb { count: left.count + right.count, root: tree};
}

// Concatenate two nodes into a single balanced branch
// Since we always return a branch, but there may be M or fewer items,
// the branch may be redundant and can be unwrapped by the caller.
fn concat_nodes<T: RrbElement>(left: &Node<T>, right: &Node<T>, top: bool) -> Branch<T> {
    // first, we handle trees of different heights
    if left.height() > right.height() {
        match left {
            Node::Branch(b) => {
                let middle = concat_nodes(b.items.last().unwrap(), right, false);
                return rebalance(Some((*b).clone()), middle, None, top);
            }
            Node::Leaf(_) => unreachable!(),
        }
    }
    if left.height() < right.height() {
        match right {
            Node::Branch(b) => {
                let middle = concat_nodes(left, b.items.first().unwrap(), false);
                return rebalance(None, middle, Some((*b).clone()), top);
            }
            Node::Leaf(_) => unreachable!(),
        }
    }

    // then, we handle leaf nodes
    if left.is_leaf() && right.is_leaf() {
        let total = left.length() + right.length();
        let left_items = match left {
            Node::Leaf(leaf) => &leaf.items,
            Node::Branch(_) => unreachable!(),
        };
        let right_items = match right {
            Node::Leaf(leaf) => &leaf.items,
            Node::Branch(_) => unreachable!(),
        };
        if top && total <= M {
            return branch(1,  vec![Node::Leaf(leaf(left_items.iter().chain(right_items.iter()).cloned().collect()))]);
        } else {
            // this may not be balanced, but the outer recursive step will rebalance it later
            return branch(1, [left.clone(), right.clone()].to_vec());
        }
    }

    // finally, we handle branches of equal height
    if left.is_branch() && right.is_branch() {
        let left_items = match left {
            Node::Branch(branch) =>  &branch.items,
            Node::Leaf(_) => unreachable!(),
        };
        let right_items = match right {
            Node::Branch(branch) => &branch.items,
            Node::Leaf(_) => unreachable!(), 
        };
        let middle = concat_nodes(
            left_items.last().unwrap(),
            right_items.first().unwrap(),
            false,
        );
        if let (&Node::Branch(ref l), &Node::Branch(ref r)) = (left, right) {
            return rebalance(Some(l.clone()), middle, Some(r.clone()), top);
        } else {
            unreachable!();
        }
    }

    unreachable!("unreachable code");
}

// create a single, balanced branch containing all items from the input branches.
fn rebalance<T: RrbElement>(left: Option<Branch<T>>, middle: Branch<T>, right: Option<Branch<T>>, top: bool) -> Branch<T> {
    // merge into a single, unbalanced node that may contain up to 2*M items
    let l = left.map_or(vec![], |n| n.items[0..n.items.len()-1].to_vec());
    let r = right.map_or(vec![], |n| n.items[1..].to_vec());
    let m = middle.items;
    let merged = branch(middle.height, l.into_iter().chain(m.into_iter()).chain(r.into_iter()).collect());
   
    // create a paln of how the items should be balanced
    let plan = create_concat_plan(&merged);
    // create a single, balanced node that may contain up to 2 * M items
    let balanced = execute_concat_plan(&merged, &plan);

    if plan.len() <= M {
        return if top == true  {
            balanced
        } else {
            branch(balanced.height+1, [Node::Branch(balanced)].to_vec())
        }
    } else {
        // distribute the(up to 2M) items across 2 nodes in a new branch
        let lbranch = branch(balanced.height, balanced.items[0..M].to_vec());
        let rbranch = branch(balanced.height, balanced.items[M..].to_vec());
        return branch(balanced.height+1, [Node::Branch(lbranch), Node::Branch(rbranch)].to_vec());
    }
}

// generate a plan of how the items in 'node' should be
// distributed that conforms to the search step invariant
fn create_concat_plan<T: RrbElement>(branch: &Branch<T>) -> Vec<usize> {
    // our initial plan is the current distribution of items
    let mut plan: Vec<usize> = branch.items.iter().map(|n|  n.length()).collect();
    // count the total number of items
    let total: usize = plan.iter().sum(); 
    // calcuate the optimal number of slots necessary
    let optimal = (total as f64 / M as f64).ceil() as usize;

    let mut i = 0;
    let mut n = plan.len();
    // first we check if our invariant, S <= [P/M] + E is met. If not, proceed.
    while n > optimal + E_MAX {
        // skip any slots with M - E/2 items as these don't need redistributing.
        // Once we reach the first slot with fewer than M - E/2 items, there will
        // always be enough space in the subsequent slots for us to distribute
        // the items over.
        while plan[i] >= M - E_MAX / 2 {
            i += 1;
        }
        // track remaining items to distribute, which starts as all the items
        // from the current slot we're going to distribute
        let mut r = plan[i];
        while r > 0 {
            // remove the slots that needs redistributing and add as many of
            // its items as posible to the next slot, then as many of the remainer
            // to the next one, and so on.
            plan[i] = cmp::min(r + plan[i+1], M);
            r = r + plan[i+1] - plan[i];
            i += 1
        }

        // slots that were distributed over were shuffled one slot to the left,
        // so we need to do the same for any remaining slot
        for j in i..n-1 {
            plan[j] = plan[j+1];
        }

        // account for shuffling slots to the left
        i -= 1;
        n -= 1;
    }

    plan[0..n].to_vec()
}

fn execute_concat_plan<T: RrbElement>(node: &Branch<T>, plan: &Vec<usize>) -> Branch<T> {
    let mut items: Vec<Node<T>> = Vec::new();

    let mut i = 0;
    let mut offset = 0;
    plan.iter().for_each(|&slot| {
        if offset == 0 && node.items[i].length() == slot {
            items.push(node.items[i].clone());
            i+=1;
        } else {
            let mut current: Vec<NodeOrValue<T>> = Vec::new();
            // hack since rust doesn't have union types
            while current.len() < slot {
                let required = slot - current.len();
                let size = node.items[i].length();
                let available = size - offset;
                let min = cmp::min(required, available);
                match &node.items[i] {
                    Node::Branch(b) => {
                        let items = &b.items[offset..offset+min];
                        for item in items {
                            current.push(NodeOrValue::Node(item.clone()));
                        }
                    }
                    Node::Leaf(l) => {
                        let items =  &l.items[offset..offset+min];
                        for item in items {
                            current.push(NodeOrValue::Value(item.clone()));
                        }}
                };
                if min == available {
                    offset = 0;
                    i += 1;
                } else {
                    offset += min;
                }
            }

            let current_values: Vec<T> = current.clone().into_iter().filter_map(|nv| {
                match nv {
                    NodeOrValue::Node(_) => { None }
                    NodeOrValue::Value(v) => { Some(v) }
                }
            }).collect();
            let current_nodes: Vec<Node<T>> = current.into_iter().filter_map(|nv| {
                match nv {
                    NodeOrValue::Node(n) => { Some(n) }
                    NodeOrValue::Value(_) => { None }
                }
            }).collect();
            if current_nodes.len() > 0 {
                items.push(Node::Branch(branch(node.height - 1, current_nodes)));
            } else {
                items.push(Node::Leaf(leaf(current_values)));
            }
        }
    });
    return branch(node.height, items);
}

#[cfg(test)]
mod tests {
    use super::*;
    use quickcheck::{Arbitrary, QuickCheck, Gen};

    #[derive(Clone, Debug)]
    struct CustomVec(Vec<usize>);

    impl Arbitrary for CustomVec {
        fn arbitrary(g: &mut Gen) -> Self {
            let size = usize::arbitrary(g) % 10 + 1;  // 1 - 10
            let mut vec = Vec::with_capacity(size);
            for _ in 0..size {
                vec.push(usize::arbitrary(g) % 1024 + 1);  // 1 - 1024
            }
            CustomVec(vec)
        }
    }

    #[test]
    fn test_branch_with_height_gt_1() {
        let sizes= [1025, 1025, 1025, 1025];
        let arrs: Vec<Vec<String>> = sizes.iter().map(|&n| { array_of_size(n as usize)}).collect();
        let rrbs = arrs.iter().map(|arr| { from_array(arr.clone())}).collect();
        let merged = concat_many(rrbs);
        assert!(assert_equal_elements(merged, arrs.into_iter().flatten().collect()));
    }

    #[test]
    fn test_distributed_slot_with_remainer() {
        let vec = from_array(array_of_size(17));

        let left = concat_many(vec![vec.clone(), vec.clone(), vec.clone(), vec.clone(), vec.clone(), vec.clone()]);
        let left_items = match left.root {
            Node::Branch(ref b) => b.items.iter().map(|i| i.length()).collect::<Vec<usize>>(),
            Node::Leaf(l) => unreachable!(),
        };
        assert_eq!(left_items, vec![17, 17, 17, 17, 17, 17]);

        let merged = concat(left, vec);
        let merged_items = match merged.root {
            Node::Branch(b) => b.items.iter().map(|i| i.length()).collect::<Vec<usize>>(),
            Node::Leaf(l) => unreachable!(),
        };
        assert_eq!(merged_items, vec![32, 19, 17, 17, 17, 17]);
    }

    #[test]
    fn order_of_elements_is_maintained() {
        fn _test_oder_of_elements_is_maintained(arr :Vec<usize>) -> bool {
            let arrs: Vec<Vec<String>> = arr.iter().map(|&i| array_of_size(i)).collect();
            let rrbs = arrs.iter().map(|a: &Vec<String>| from_array(a.clone())).collect();
            let merged = concat_many(rrbs);
            let arrs_flat = arrs.into_iter().flatten().collect();
            let res = assert_equal_elements(merged, arrs_flat);
            res
        }

        fn prop(v: CustomVec) -> bool {
            _test_oder_of_elements_is_maintained(v.0)
        }
        QuickCheck::new().tests(100).quickcheck(prop as fn(CustomVec) -> bool);
    }

    #[test]
    fn search_step_invariant_is_maintained() {
        fn _test_search_step_invariant_is_maintained(arr: Vec<usize>) -> bool {
            let arrs: Vec<Vec<String>> = arr.iter().map(|&i| array_of_size(i)).collect();
            let rrbs = arrs.iter().map(|a: &Vec<String>| from_array(a.clone())).collect();
            let merged = concat_many(rrbs);
            let res = assert_search_step_invariant(&merged.root);
            res
        }

        fn prop(v: CustomVec) -> bool {
            _test_search_step_invariant_is_maintained(v.0)
        }

        QuickCheck::new().tests(100).quickcheck(prop as fn(CustomVec) -> bool);
    }

    #[test]
    fn height_invariant_is_maintained() {
        fn _test_search_step_invariant_is_maintained(arr: Vec<usize>) -> bool {
            let arrs = arr.iter().map(|&i| array_of_size(i)).collect::<Vec<Vec<String>>>();
            let rrbs = arrs.iter().map(|a: &Vec<String>| from_array(a.clone())).collect();
            let merged = concat_many(rrbs);

            let length= arrs.into_iter().flatten().collect::<Vec<String>>().len();
            let height_least_dense =  if length > 0 {
                ((length as f64).ln() / ((M - E_MAX) as f64).ln()) as usize
             } else {
                0
             };
             let height_most_dense = if length > 0 {
            ((length as f64).ln() / (M as f64).ln() - 1.0).ceil() as usize
             } else {
                0
             };
             let res = merged.root.height() <= height_least_dense && merged.root.height() >= height_most_dense;
            res
        }

        fn prop(v: CustomVec) -> bool {
            _test_search_step_invariant_is_maintained(v.0)
        }

        QuickCheck::new().tests(100).quickcheck(prop as fn(CustomVec) -> bool);
    }

    fn array_of_size(size: usize) -> Vec<String> {
        (0..size).map(|i| {base26(i)}).collect()
    }

    fn base26(i: usize) -> String {
        if i == 0 {
            return "A".to_string();
        }
        
        let mut result = String::new();
        let mut num = i;
        
        while num > 0 {
            num -= 1; // Adjust to 0-based (A=0, Z=25)
            let remainder = num % 26;
            let char_code = (remainder as u8 + b'A') as char;
            result.insert(0, char_code);
            num /= 26;
        }

        result
    }

    fn concat_many<T: RrbElement> (rrbs: Vec<Rrb<T>>) -> Rrb<T> { 
        if rrbs.len() == 0 {
            return init();
        }
        let first = rrbs[0].clone();
        let output = rrbs.into_iter().skip(1).fold(first, |acc, rrb| {
            concat(acc, rrb)
        });
        output
    }

    fn assert_equal_elements<T: RrbElement>(rrb: Rrb<T>, v: Vec<T>) -> bool {
        if rrb.count !=  v.len() {
            return false
        }
        for (idx, item) in v.iter().enumerate() {
            if &get(&rrb, idx).unwrap() != item {
                return false
            }
        }
        true
    }

    fn assert_search_step_invariant<T: RrbElement>(node: &Node<T>) -> bool {
        match node {
            Node::Leaf(_) => true,
            Node::Branch(b) => {
                let s = b.items.iter().fold(0, |acc, branch| { acc + branch.length()});
                let opt = (s as f64 / M as f64).ceil() as usize;
                let limit = opt + E_MAX;
                if node.length() > limit {
                    return false
                }
                if node.height() > 1 {
                    for ref i in b.items.clone() {
                        if !assert_search_step_invariant(i) {
                            return false
                        }
                    }
                }
                true
            }
        }
    }
}