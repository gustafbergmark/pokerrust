use serde::{Deserialize, Serialize};
use std::ops::Index;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CombinationMap<T: Sized + Clone, const N: usize, const K: usize> {
    values: Vec<Option<T>>,
}

impl<T: Sized + Clone, const N: usize, const K: usize> CombinationMap<T, N, K> {
    pub fn new() -> Self {
        assert!(N <= 64);
        assert!(K <= N);
        Self {
            values: vec![None; choose(N, K)],
        }
    }

    pub fn get(&self, key: u64) -> Option<&T> {
        match self.get_ordering_index(key) {
            None => None,
            Some(index) => self.values[index].as_ref(),
        }
    }

    pub fn get_mut(&mut self, key: u64) -> Option<&mut T> {
        match self.get_ordering_index(key) {
            None => None,
            Some(index) => self.values[index].as_mut(),
        }
    }

    pub fn insert(&mut self, key: u64, value: T) -> Option<T> {
        let index = self
            .get_ordering_index(key)
            .expect("Key is not valid combination");
        let old = self.get(key).map(|elem| elem.clone());
        self.values[index] = Some(value);
        old
    }

    #[inline]
    fn get_ordering_index(&self, mut set: u64) -> Option<usize> {
        if set.count_ones() as usize == K {
            let mut res = 0;
            for c in 1..=K {
                let i = set.trailing_zeros();
                res += choose(i as usize, c);
                set ^= 1 << i;
            }
            Some(res)
        } else {
            None
        }
    }

    #[inline]
    fn next(x: u64) -> u64 {
        let c = x & -(x as i64) as u64;
        let r = x + c;
        let ret = ((x ^ r) / (4 * c)) | r;
        assert_eq!(ret.count_ones(), x.count_ones());
        ret
    }
}

#[inline]
pub fn choose(n: usize, k: usize) -> usize {
    (n - k + 1..=n).product::<usize>() / (1..=k).product::<usize>()
}

impl<T: Sized + Clone, const N: usize, const K: usize> Index<u64> for CombinationMap<T, N, K> {
    type Output = T;

    fn index(&self, index: u64) -> &Self::Output {
        &self.get(index).expect("Key not in map")
    }
}
//
// impl<T: Sized + Clone, const N: usize, const K: usize> IntoIterator for CombinationMap<T, N, K> {
//     type Item = (u64, T);
//     type IntoIter = Iterator<Item = Self::Item>;
//
//     fn into_iter(self) -> Self::IntoIter {
//         let mut key = 1 << K - 1;
//         let mut res = Vec::new();
//         for elem in self.values {
//             res.push((key, elem));
//             key = Self::next(key);
//         }
//         res.into_iter()
//     }
// }
//
// pub struct CombinationMapIterator<T: Sized + Clone, const N: usize, const K: usize> {
//     map: CombinationMap<T, N, K>,
//     index: u64,
// }
//
// impl<T: Sized + Clone, const N: usize, const K: usize> Iterator
//     for CombinationMapIterator<T, N, K>
// {
//     type Item = (u64,T);
//     fn next(&mut self) -> Option<Self::Item> {
//
//     }
// }
