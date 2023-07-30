use std::ops::{Add, AddAssign, Div, DivAssign, Index, IndexMut, Mul, MulAssign, Sub, SubAssign};

#[repr(align(256))]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Vector {
    pub values: [f64; 1326],
}
#[allow(unused)]
impl Vector {
    pub fn from(v: &[f64]) -> Self {
        Vector {
            values: v.try_into().unwrap(),
        }
    }
    pub fn ones() -> Self {
        Vector {
            values: [1.0; 1326],
        }
    }

    pub fn sum(&self) -> f64 {
        self.values.iter().sum::<f64>()
    }

    pub fn norm(&self) -> Self {
        let n = self.sum();
        if n == 0.0 {
            return Vector::default();
        } else {
            *self / n
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
    type Output = f64;

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
            values[i] = self.values[i] * rhs.values[i];
        }
        Vector { values }
    }
}

impl Mul<f64> for Vector {
    type Output = Self;

    fn mul(self, rhs: f64) -> Self::Output {
        let mut values = [0.0; 1326];
        for i in 0..1326 {
            values[i] = self.values[i] * rhs;
        }
        Vector { values }
    }
}

impl MulAssign for Vector {
    fn mul_assign(&mut self, rhs: Self) {
        for i in 0..1326 {
            self.values[i] *= rhs.values[i];
        }
    }
}

impl MulAssign<f64> for Vector {
    fn mul_assign(&mut self, rhs: f64) {
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
            values[i] = self.values[i] + rhs.values[i];
        }
        Vector { values }
    }
}

impl AddAssign for Vector {
    fn add_assign(&mut self, rhs: Self) {
        for i in 0..1326 {
            self.values[i] += rhs.values[i];
        }
    }
}

impl Sub for Vector {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        let mut values = [0.0; 1326];
        for i in 0..1326 {
            values[i] = self.values[i] - rhs.values[i];
        }
        Vector { values }
    }
}

impl SubAssign for Vector {
    fn sub_assign(&mut self, rhs: Self) {
        for i in 0..1326 {
            self.values[i] -= rhs.values[i];
        }
    }
}

impl Div for Vector {
    type Output = Self;

    fn div(self, rhs: Self) -> Self::Output {
        let mut values = [0.0; 1326];
        for i in 0..1326 {
            values[i] = self.values[i] / rhs.values[i];
        }
        Vector { values }
    }
}

impl DivAssign for Vector {
    fn div_assign(&mut self, rhs: Self) {
        for i in 0..1326 {
            self.values[i] /= rhs.values[i];
        }
        for i in 0..1326 {
            if self.values[i].is_nan() {
                self.values[i] = 0.5;
            }
        }
    }
}
impl Div<f64> for Vector {
    type Output = Self;

    fn div(self, rhs: f64) -> Self::Output {
        let mut values = [0.0; 1326];
        for i in 0..1326 {
            values[i] = self.values[i] / rhs;
        }
        Vector { values }
    }
}
