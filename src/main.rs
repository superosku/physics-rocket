use minifb::{MouseMode, Key, Window, WindowOptions};
use raqote::{DrawTarget, SolidSource, Source, DrawOptions, PathBuilder, StrokeStyle, LineCap, LineJoin};
use rand::Rng;
use rayon::prelude::*;
use crate::neural::neural_net;
use crate::point::Vector;

mod point;
mod neural;

const WIDTH: usize = 1000;
const HEIGHT: usize = 800;

const ITERATIONS: usize = 500;

#[derive(Clone)]
struct Ship {
    pos1: point::Vector,
    pos2: point::Vector,
    pos_1_last: point::Vector,
    pos_2_last: point::Vector,
    // vel1: point::Vector,
    // vel2: point::Vector,
    angle1: f32,
    angle2: f32,
    throttle1: f32,
    throttle2: f32,

    best_distance: Option<f32>,
    score: f32,
    dead: bool,

    neural_net: neural_net,
}

fn world_to_screen(point: point::Vector) -> point::Vector {
    point::Vector::new(
        point.x + (WIDTH / 2) as f32,
        point.y + (HEIGHT / 2) as f32
    )
}

fn screen_to_world(point: point::Vector) -> point::Vector {
    point::Vector::new(
        (point.x - (WIDTH / 2) as f32) / 100.,
         (point.y - (HEIGHT / 2) as f32) / 100.
    )
}

impl Ship {
    fn new() -> Ship {
        let mut rng = rand::thread_rng();
        // let angle1: f32 = rng.gen::<f32>() - 0.5;
        // let angle2: f32 = rng.gen::<f32>() - 0.5;
        // let throttle1: f32 = rng.gen::<f32>() * 0.8 + 0.0;
        // let throttle2: f32 = rng.gen::<f32>() * 0.8 + 0.0;
        // let throttle1: f32 = 0.0;

        Ship {
            pos1: point::Vector::new(0.5, 0.),
            pos2: point::Vector::new(-0.5, 0.),
            pos_1_last: point::Vector::new(0.5, 0.),
            pos_2_last: point::Vector::new(-0.5, 0.),
            angle1: 0.,
            angle2: 0.,
            throttle1: 0.,
            throttle2: 0.,
            score: 0.,
            best_distance: None,
            dead: false,
            // ship_angle, x_dist, y_dist, ship_angle_velocity, ship_velocity_x, ship_velocity_y
            neural_net: neural::neural_net::new(vec![6 + 4, 4])
        }
    }

    fn new_from_two(first: &Ship, second: &Ship) -> Ship {
        let mut new_ship = first.clone();
        new_ship.neural_net = new_ship.neural_net.mix_randomly_with_other(&second.neural_net);
        new_ship
    }

    fn clone_for_mutation(&self, lr: f32) -> Ship {
        let mut new_ship = Ship::new();
        new_ship.neural_net = self.neural_net.clone_mutated(lr);
        new_ship
    }

    fn reset(&mut self, spread: f32) {
        // Randomize the starting position
        let mut rng = rand::thread_rng();
        let xdiff: f32 = (rng.gen::<f32>() - 0.5) * spread;
        let ydiff: f32 = (rng.gen::<f32>() - 0.5) * spread;
        // let xdiff: f32 = 2.;
        // let ydiff: f32 = 2.;

        self.pos1 = point::Vector::new(0.5 + xdiff, 0. + ydiff);
        self.pos2 = point::Vector::new(-0.5 + xdiff, 0. + ydiff);
        self.pos_1_last = point::Vector::new(0.5 + xdiff, 0. + ydiff);
        self.pos_2_last = point::Vector::new(-0.5 + xdiff, 0. + ydiff);
        self.angle1 = 0.;
        self.angle2 = 0.;
        self.throttle1 = 0.;
        self.throttle2 = 0.;
        self.dead = false;
        self.score = 0.;
        self.best_distance = None;
    }

    fn draw_motor(&self, dt: &mut DrawTarget, point: & point::Vector, angle: f32, throttle: f32) {
        let mut pb = PathBuilder::new();

        let center = point::Vector::new(point.x + (WIDTH / 2) as f32, point.y + (HEIGHT / 2) as f32);

        let side= point::Vector::new(angle.cos(), angle.sin());
        let forward = point::Vector::new(angle.sin(), -angle.cos());

        let width = 10.;
        let height = 20.;

        let top_left = center.added(&forward.multiplied(height)).added(&side.multiplied(width));
        let top_right = center.added(&forward.multiplied(height)).added(&side.multiplied(-width));
        let bottom_left = center.added(&forward.multiplied(-height)).added(&side.multiplied(width));
        let bottom_right = center.added(&forward.multiplied(-height)).added(&side.multiplied(-width));

        pb.move_to(bottom_left.x, bottom_left.y);
        pb.line_to(top_left.x, top_left.y);
        pb.line_to(top_right.x, top_right.y);
        pb.line_to(bottom_right.x, bottom_right.y);
        // pb.close();

        let path = pb.finish();
        dt.fill(
            &path,
            &Source::Solid(SolidSource::from_unpremultiplied_argb(0xff, 0xbb, 0xbb, 0xbb)),
            &DrawOptions::new()
        );
        dt.stroke(
            &path,
            &Source::Solid(SolidSource::from_unpremultiplied_argb(0xff, 0x66, 0x66, 0x66)),
            &StrokeStyle{
                cap: LineCap::Butt,
                join: LineJoin::Miter,
                width: 4.,
                miter_limit: 1.,
                dash_array: vec![],
                dash_offset: 0.,
            },
            &DrawOptions::new()
        );


        let fire_end_1 = center.added(&forward.multiplied(-height - throttle * 30.));
        let fire_end_2 = center.added(&forward.multiplied(-height - throttle * 15.));

        {
            let mut pb_fire = PathBuilder::new();
            pb_fire.move_to(bottom_left.x, bottom_left.y);
            pb_fire.line_to(bottom_right.x, bottom_right.y);
            pb_fire.line_to(fire_end_1.x, fire_end_1.y);
            let path = pb_fire.finish();
            dt.fill(
                &path,
                &Source::Solid(SolidSource::from_unpremultiplied_argb(0xff, 0xbb, 0xbb, 0x00)),
                &DrawOptions::new()
            );
        }
        {
            let mut pb_fire = PathBuilder::new();
            pb_fire.move_to(bottom_left.x, bottom_left.y);
            pb_fire.line_to(bottom_right.x, bottom_right.y);
            pb_fire.line_to(fire_end_2.x, fire_end_2.y);
            let path = pb_fire.finish();
            dt.fill(
                &path,
                &Source::Solid(SolidSource::from_unpremultiplied_argb(0xff, 0xbb, 0x00, 0x00)),
                &DrawOptions::new()
            );
        }
    }

    fn draw(&self, dt: &mut DrawTarget) {
        if self.dead {
            return;
        }
        let camera_pos_1 = self.pos1.multiplied(100.);
        let camera_pos_2 = self.pos2.multiplied(100.);

        let ship_normal = self.pos1.subtracted(&self.pos2).normalized();
        let ship_angle = ship_normal.angle() - std::f32::consts::PI / 2.;

        self.draw_motor(dt, &camera_pos_1, self.angle1 - ship_angle, self.throttle1);
        self.draw_motor(dt, &camera_pos_2, self.angle2 - ship_angle, self.throttle2);

        let mut pb = PathBuilder::new();
        pb.move_to(camera_pos_1.x + (WIDTH / 2) as f32, camera_pos_1.y + (HEIGHT / 2) as f32);
        pb.line_to(camera_pos_2.x + (WIDTH / 2) as f32, camera_pos_2.y + (HEIGHT / 2) as f32);

        let path = pb.finish();

        dt.stroke(
            &path,
            &Source::Solid(SolidSource::from_unpremultiplied_argb(0xff, 0x55, 0x55, 0x55)),
            &StrokeStyle{
                cap: LineCap::Butt,
                join: LineJoin::Miter,
                width: 10.,
                miter_limit: 1.,
                dash_array: vec![],
                dash_offset: 0.,
            },
            &DrawOptions::new()
        )
    }

    fn do_brain(&mut self, goal: &point::Vector) {
        if self.dead {
            return;
        }
        // ship_angle, x_dist, y_dist, ship_angle_velocity, ship_velocity_x, ship_velocity_y
        let ship_angle = self.pos1.subtracted(&self.pos2).normalized().angle();
        let ship_old_angle = self.pos_1_last.subtracted(&self.pos_2_last).normalized().angle();
        let ship_angle_velocity = ship_angle - ship_old_angle;

        let ship_center = self.pos1.added(&self.pos2).multiplied(0.5);
        let ship_center_last = self.pos_1_last.added(&self.pos_2_last).multiplied(0.5);
        let ship_velocity_x = ship_center.x - ship_center_last.x;
        let ship_velocity_y = ship_center.y - ship_center_last.y;

        let vector_to_goal = goal.subtracted(&ship_center);

        let x_dist = vector_to_goal.x;
        let y_dist = vector_to_goal.y;

        let last_layer = self.neural_net.get_last_layer();

        self.neural_net.set_first_layer(vec![
            ship_angle,
            x_dist,
            y_dist,
            ship_angle_velocity,
            ship_velocity_x,
            ship_velocity_y,
            // Last layer that is used
            0., //last_layer[0],
            0., //last_layer[1],
            0., //last_layer[2],
            0., //last_layer[3],
            // Extras from last layer
            // last_layer[4],
            // last_layer[5],
            // last_layer[6],
        ]);
        self.neural_net.forward_propagate();
        let output = self.neural_net.get_last_layer();

        self.throttle1 = output[0] * 1.0;
        self.throttle2 = output[1] * 1.0;
        self.angle1 = (output[2] - 0.5) * 2.0;
        self.angle2 = (output[3] - 0.5) * 2.0;
        // self.angle1 = 0.0;
        // self.angle2 = 0.0;
        // Print debug angles and throttles
        // println!("Angle1: {}, Angle2: {}, Throttle1: {}, Throttle2: {}", self.angle1, self.angle2, self.throttle1, self.throttle2);
    }

    fn update_score(&mut self) {
        let middle = self.pos1.added(&self.pos2).multiplied(0.5);
        let x_dist = middle.x;
        let y_dist = middle.y;
        let distance = (x_dist * x_dist + y_dist * y_dist).sqrt();
        let distance_score = (distance * 0.5).powf(2.) * 1.;

        match self.best_distance {
            None => {
                self.best_distance = Some(distance);
            },
            Some(best) => {
                if distance < best {
                    self.best_distance = Some(distance);
                    self.score -= best - distance;
                } else {
                    self.score += distance - best;
                }
            }
        }

        let speed_x = self.pos1.x - self.pos_1_last.x;
        let speed_y = self.pos1.y - self.pos_1_last.y;
        let speed = (speed_x * speed_x + speed_y * speed_y).sqrt();
        let speed_pow = ((speed + 1.0) * 10.).powf(2.);
        let speed_score = 0.;

        // self.score += distance_score + speed_score;
    }

    fn simulate(&mut self) {
        if self.dead {
            return;
        }
        // Store current position for storing later
        let temp1 = self.pos1.clone();
        let temp2 = self.pos2.clone();

        let ship_normal = self.pos1.subtracted(&self.pos2).normalized();
        let ship_angle = ship_normal.angle() - std::f32::consts::PI / 2.;

        let real_angle_1 = self.angle1 - ship_angle;
        let real_angle_2 = self.angle2 - ship_angle;

        // Do verlet stuff
        self.pos1.add(
            // Current position
            &self.pos1
            // Minus old position
            .added(&self.pos_1_last.multiplied(-1.))
            // Add gravity
            .added(&point::Vector::new(0., 0.002))
            // Add throttle
            .added(
                &point::Vector::new(
                    real_angle_1.sin(),
                    -real_angle_1.cos()
                ).multiplied(self.throttle1 * 0.005)
            )
        );

        self.pos2.add(
            // Current position
            &self.pos2
            // Minus old position
            .added(&self.pos_2_last.multiplied(-1.))
            // Add gravity
            .added(&point::Vector::new(0., 0.002))
            // Add throttle
            .added(
                &point::Vector::new(
                    real_angle_2.sin(),
                    -real_angle_2.cos()
                ).multiplied(self.throttle2 * 0.005)
            )
        );

        // Make sure distance between points stays 1
        let direction = self.pos2.added(&self.pos1.negated());
        let distance = direction.length();
        self.pos1.add(
            &direction.multiplied(
                (distance - 1.0) / distance * 0.5
            )
        );
        self.pos2.add(
            &direction.multiplied(
                (distance - 1.0) / distance * 0.5
            ).negated()
        );

        // Save old position from temp
        self.pos_1_last.x = temp1.x;
        self.pos_1_last.y = temp1.y;
        self.pos_2_last.x = temp2.x;
        self.pos_2_last.y = temp2.y;

        // Set to dead if out of bounds
        if self.pos1.length() > 10.0 {
            self.dead = true;
        }
    }
}

fn do_ship_mutation(ships: &mut Vec<Ship>, spread: f32, lr: f32) {
    ships.sort_by(|a, b| a.score.partial_cmp(&b.score).unwrap());

    let mut new_ships:Vec<Ship> = vec![];

    for i in 0..(ships.len() / 2) {
        let ship = ships[i].clone();
        new_ships.push(ship);
    }

    while new_ships.len() < ships.len() {
        let mut rng = rand::thread_rng();
        let random_ship = ships[(rng.gen::<f32>().powf(2.0) * ships.len() as f32) as usize].clone_for_mutation(lr);
        if rng.gen::<f32>() < 0.5 {
            let random_ship_2 = ships[(rng.gen::<f32>().powf(2.0) * ships.len() as f32) as usize].clone_for_mutation(lr);
            new_ships.push(Ship::new_from_two(&random_ship, &random_ship_2))
        } else {
            new_ships.push(random_ship);
        }
    }

    for i in 0..ships.len() {
        ships[i] = new_ships[i].clone();
    }

    // // Reset ships position
    for ship in &mut *ships {
        ship.reset(spread);
    }
}

fn iterate_draw(ships: &mut Vec<Ship>, steps: i32, spread: f32, lr: f32) {
    let mut window = Window::new(
        "Raqote",
        WIDTH,
        HEIGHT,
        WindowOptions { ..WindowOptions::default() },
    ).unwrap();

    // let font = SystemSource::new()
    //     .select_best_match(&[FamilyName::SansSerif], &Properties::new())
    //     .unwrap()
    //     .load()
    //     .unwrap();

    let size = window.get_size();
    let mut dt = DrawTarget::new(size.0 as i32, size.1 as i32);

    window.limit_update_rate(Some(std::time::Duration::from_micros(16600)));

    let mut iteration = 0;
    // Max value of f32
    let mut bestaverage_score: f32 = std::f32::MAX;

    // screen_to_world
    let mut mouse_pos_world = point::Vector::new(0., 0.);

    while window.is_open() && !window.is_key_down(Key::Escape) {
        iteration += 1;
        dt.clear(SolidSource::from_unpremultiplied_argb(0xff, 0x00, 0x00, 0x00));

        if let Some(pos) = window.get_mouse_pos(MouseMode::Clamp) {
            // mouse_x = pos.0 as i32;
            // mouse_y = pos.1 as i32;

            mouse_pos_world = screen_to_world(
                point::Vector::new(pos.0 as f32, pos.1 as f32),
            );
        }


        // Iterate each ship
        for ship in &mut *ships {
            ship.do_brain(&mouse_pos_world);
            ship.simulate();
            ship.update_score();
            ship.draw(&mut dt);
        }

        if iteration == steps {
            // Get average score of ships
            let mut average_score = 0.;
            for ship in &mut *ships {
                average_score += ship.score;
            }
            average_score /= ships.len() as f32;
            bestaverage_score = bestaverage_score.min(average_score);

            ships.sort_by(|a, b| a.score.partial_cmp(&b.score).unwrap());
            let best_scores = ships.iter().map(|ship| ship.score).take(8).collect::<Vec<f32>>();
            let best_score_string = best_scores.iter().map(|score| score.to_string()).collect::<Vec<String>>().join(" ");
            // println!("Average score: {} {} {} {} [{}]", bestaverage_score, spread, lr, average_score, best_score_string);

            do_ship_mutation(ships, 0., lr);

            iteration = 0;
        }

        window.update_with_buffer(dt.get_data(), size.0, size.1).unwrap();
    }
}

fn iterate_raw(ships: &mut Vec<Ship>, steps: i32, spread: f32, lr: f32, step_n: i32, iteration_n: i32) -> f32 {
    let THREAD_COUNT: usize = 16;
    let vec: Vec<i64> = (0..(THREAD_COUNT as i64)).collect();

    // Shuffle ships to get better balance for threads
    let mut rng = rand::thread_rng();
    for i in 0..ships.len() {
        // let random_index = (rng.gen::<f32>().powf(2.0) * ships.len() as f32) as usize;
        let random_index = (rng.gen::<f32>() * ships.len() as f32) as usize;
        let temp = ships[i].clone();
        ships[i] = ships[random_index].clone();
        ships[random_index] = temp;
    }

    // Do stuff in threads
    let collected_ships : Vec<Vec<Ship>> = vec.par_iter().map(
        |i| {
        let mut splitted_ships:Vec::<Ship> = Vec::new();
        for j in 0..ships.len() {
            if j % THREAD_COUNT == *i as usize {
                splitted_ships.push(ships[j].clone());
            }
        }

        // Random goal based on spread
        let mut goal = point::Vector::new(0., 0.);
        // goal.x = rng.gen::<f32>() * spread - spread / 2.;
        // goal.y = rng.gen::<f32>() * spread - spread / 2.;

        for step_n in 0..(steps as usize) {
            // Goal is a unit circle
            let direction = if iteration_n % 2 == 0 { 1. } else { -1. };
            goal = point::Vector::new(
                (direction * step_n as f32 / steps as f32 * 2. * std::f32::consts::PI * 10.).sin() * spread,
                (direction * step_n as f32 / steps as f32 * 2. * std::f32::consts::PI * 10.).cos() * spread,
            );

            for ship in &mut splitted_ships {
                ship.do_brain(&goal);
                ship.simulate();
                ship.update_score();
            }
        }
        splitted_ships
    }).collect();

    // Collect ships back from threads
    ships.clear();
    for collected_ship in collected_ships {
        for ship in collected_ship {
            ships.push(ship);
        }
    }

    // Get average score of ships
    let mut average_score = 0.;
    for ship in &mut * ships {
        average_score += ship.score;
    }
    average_score /= ships.len() as f32;

    ships.sort_by(|a, b| a.score.partial_cmp(&b.score).unwrap());
    let best_scores = ships.iter().map(|ship| ship.score).take(8).collect::<Vec<f32>>();
    let best_score_string = best_scores.iter().map(|score| score.to_string()).collect::<Vec<String>>().join(" ");
    // println!("Average score: {} [{}]", average_score, best_score_string);

    // Get scores of top 10% of ships
    let best_scores_of_10_percent = ships.iter().map(|ship| ship.score).take((ships.len() as f32 * 0.1) as usize).collect::<Vec<f32>>();
    let average_of_top_10_percent = best_scores_of_10_percent.iter().sum::<f32>() / best_scores_of_10_percent.len() as f32;

    println!("Average score: {} {} {} {} [{}] {}", step_n, lr, average_of_top_10_percent, average_score, best_score_string, spread);

    do_ship_mutation(ships, 0., lr);

    average_score
}

fn main() {
    // Vector of ships
    let mut ships: Vec<Ship> = Vec::new();
    // Add 10 ships to vector
    for _ in 0..1000 {
        let mut ship = Ship::new();
        ship.reset(0.);
        ships.push(ship);
    }

    // let mut spread: f32 = 0.;
    // let mut spread: f32 = 3.;
    // let mut spread: f32 = 0.;
    let mut spread: f32 = 1.;

    let mut lr: f32 = 0.05;
    let mut steps = 5000;
    let mut step_n = 0;
    // let spread: f32 = 0.;
    for i in 0..100 {
        steps = 200 + i * 10;
        // spread = ((i as f32 - 100.).max(0.) / 1000.).min(4.);
        // spread = 3.;
        // if (i < 50) {
        //     lr = 0.02
        // } else if i < 200 {
        //     lr = 0.01
        // } else if i < 500 {
        //     lr = 0.005
        // } else {
        //     lr = 0.002
        // }
        iterate_raw(&mut ships, steps, spread, lr, step_n, i);
        step_n += 1;
    }

    steps = 1000;

    // Truncate ships to 50
    ships.truncate(10);

    iterate_draw(&mut ships, steps, spread, lr);
}
