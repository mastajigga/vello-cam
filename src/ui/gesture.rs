//! One active pointer, activation on release over the original target only.
#[derive(Default)]
pub struct Gesture {
    pub pointer: Option<(i32, usize)>,
    pub inside: bool,
}

impl Gesture {
    pub fn begin(&mut self, pointer: i32, target: Option<usize>) -> Option<usize> {
        if self.pointer.is_some() { return None; }
        let target = target?;
        self.pointer = Some((pointer, target));
        self.inside = true;
        Some(target)
    }

    pub fn update(&mut self, pointer: i32, target: Option<usize>) {
        if let Some((active, original)) = self.pointer {
            if active == pointer { self.inside = target == Some(original); }
        }
    }

    pub fn finish(&mut self, pointer: i32, target: Option<usize>, cancel: bool) -> Option<usize> {
        let (active, original) = self.pointer?;
        if active != pointer { return None; }
        self.clear();
        if !cancel && target == Some(original) { Some(original) } else { None }
    }

    pub fn clear(&mut self) {
        self.pointer = None;
        self.inside = false;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tap_activates_once_on_release() {
        let mut gesture = Gesture::default();
        assert_eq!(gesture.begin(7, Some(2)), Some(2));
        assert!(gesture.inside);
        assert_eq!(gesture.finish(7, Some(2), false), Some(2));
        assert_eq!(gesture.finish(7, Some(2), false), None);
    }

    #[test]
    fn drag_out_cancels_but_return_to_original_can_activate() {
        let mut g = Gesture::default();
        g.begin(1, Some(2));
        g.update(1, Some(3));
        assert!(!g.inside);
        assert_eq!(g.finish(1, Some(3), false), None);
        g.begin(1, Some(2));
        g.update(1, None);
        g.update(1, Some(2));
        assert!(g.inside);
        assert_eq!(g.finish(1, Some(2), false), Some(2));
    }

    #[test]
    fn cancellation_and_second_finger_never_activate() {
        let mut g = Gesture::default();
        assert_eq!(g.begin(1, None), None);
        g.begin(1, Some(6));
        assert_eq!(g.begin(2, Some(7)), None);
        assert_eq!(g.finish(2, Some(6), false), None);
        assert_eq!(g.pointer, Some((1, 6)));
        assert_eq!(g.finish(1, Some(6), true), None);
        g.begin(1, Some(6));
        g.clear(); // blur / resize
        assert_eq!(g.finish(1, Some(6), false), None);
    }
}
