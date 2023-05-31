use std::ops::{Add, AddAssign, Index, IndexMut, Mul, MulAssign, Sub, SubAssign};

#[derive(Clone, Copy, Debug)]
pub struct Vector {
    pub values: [f32; 1326],
}
#[allow(unused)]
impl Vector {
    pub fn from(v: &[f32]) -> Self {
        Vector {
            values: v.try_into().unwrap(),
        }
    }
    pub fn ones() -> Self {
        Vector {
            values: [1.0; 1326],
        }
    }
}

impl Default for Vector {
    fn default() -> Self {
        Vector {
            values: [0.0; 1326],
        }
    }
}

impl Index<usize> for Vector {
    type Output = f32;

    fn index(&self, index: usize) -> &Self::Output {
        &self.values[index]
    }
}

impl IndexMut<usize> for Vector {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self.values[index]
    }
}

impl Mul for Vector {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self::Output {
        let mut values = [0.0; 1326];
        for i in 0..1326 {
            values[i] = &self.values[i] * &rhs.values[i];
        }
        Vector { values }
    }
}

impl MulAssign for Vector {
    fn mul_assign(&mut self, rhs: Self) {
        for i in 0..1326 {
            self.values[i] *= &rhs.values[i];
        }
    }
}

impl MulAssign<f32> for Vector {
    fn mul_assign(&mut self, rhs: f32) {
        for i in 0..1326 {
            self.values[i] *= rhs;
        }
    }
}

impl Add for Vector {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        let mut values = [0.0; 1326];
        for i in 0..1326 {
            values[i] = &self.values[i] + &rhs.values[i];
        }
        Vector { values }
    }
}

impl AddAssign for Vector {
    fn add_assign(&mut self, rhs: Self) {
        for i in 0..1326 {
            self.values[i] += &rhs.values[i];
        }
    }
}

impl Sub for Vector {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        let mut values = [0.0; 1326];
        for i in 0..1326 {
            values[i] = &self.values[i] - &rhs.values[i];
        }
        Vector { values }
    }
}

impl SubAssign for Vector {
    fn sub_assign(&mut self, rhs: Self) {
        for i in 0..1326 {
            self.values[i] -= &rhs.values[i];
        }
    }
}
