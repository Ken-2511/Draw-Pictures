# This programis used to generate strokes from pictures.
# It uses the pre-configured picture to generate strokes.

import torch
from PIL import Image
from torchvision import transforms as trans


def load_image(f_name):
    image = Image.open(f_name)
    transform = trans.ToTensor()
    image = Image.open(f_name)
    image = transform(image)
    return image


def save_image(image, f_name):
    transform = trans.ToPILImage()
    image = transform(image)
    image.save(f_name)


def get_next_point(past_points, step_length, image):
    # get the next point by the past points
    # the past points is a list of points
    # the last point is the current point
    # the second last point is the last point
    # the third last point is the second last point
    # ...
    # based on the five last points, estimate the next point by calculating 
    # the average velocity and the average acceleration
    # but if the length of the past points is less than 2, the angle would be 360 degree
    def get_point_score(x, y, anticipated_point, last_point, image):
        # return the score by the following rules
        # 1. whether the corresponding point is bright in the image. (should be bright)
        # 2. the distance between the last point and the (x, y). (should be close to step_length)
        # 3. the distance between the anticipated point and the (x, y). (should be close to 0)
        score = 0
        if x < 0 or x >= image.shape[1] or y < 0 or y >= image.shape[2]:
            return 0
        if image[0][x][y] < 0.1:
            return 0
        score = image[0][x][y] * 10
        distance = ((x - last_point[0])**2 + (y - last_point[1])**2)**0.5
        if distance < 2:
            return 0
        score += 1 / (1 + abs(distance - step_length))
        distance = ((x - anticipated_point[0])**2 + (y - anticipated_point[1])**2)**0.5
        score += 1 / (1 + abs(distance))
        return score
    
    def choose_best_point(anticipated_point, last_point, image, scope=2):
        # choose the best point from the anticipated point
        # the best point is the point with the highest score
        # if the best point is not found, return None
        best_score = 0
        best_point = None
        for x in range(int(anticipated_point[0] - scope), int(anticipated_point[0] + scope + 1)):
            for y in range(int(anticipated_point[1] - scope), int(anticipated_point[1] + scope + 1)):
                score = get_point_score(x, y, anticipated_point, last_point, image)
                if score > best_score:
                    best_score = score
                    best_point = (x, y)
        return best_point
    
    assert len(past_points) > 0
    if len(past_points) == 1:
        return choose_best_point(past_points[-1], past_points[-1], image, scope=15)
    elif len(past_points) == 2:
        point = choose_best_point(past_points[-1], past_points[-2], image)
        return point
    else:
        if len(past_points) > 5:
            idx = len(past_points) - 5
        else:
            idx = 1
        velocities = []
        for i in range(idx, len(past_points)):
            velocities.append((past_points[i][0] - past_points[i-1][0], past_points[i][1] - past_points[i-1][1]))
        accelerations = []
        for i in range(1, len(velocities)):
            accelerations.append((velocities[i][0] - velocities[i-1][0], velocities[i][1] - velocities[i-1][1]))
        # the average velocity
        average_velocity = sum([i[0] for i in velocities]) / len(velocities), sum([i[1] for i in velocities]) / len(velocities)
        # the average acceleration
        average_acceleration = sum([i[0] for i in accelerations]) / len(accelerations), sum([i[1] for i in accelerations]) / len(accelerations)
        # the anticipated next point
        next_point = (past_points[-1][0] + average_velocity[0] + average_acceleration[0],
                      past_points[-1][1] + average_velocity[1] + average_acceleration[1])
        # makesure that the length is step_length
        dx, dy = next_point[0] - past_points[-1][0], next_point[1] - past_points[-1][1]
        length = (dx**2 + dy**2)**0.5
        next_point = (past_points[-1][0] + dx / length * step_length, past_points[-1][1] + dy / length * step_length)
        # choose the best point
        next_point = choose_best_point(next_point, past_points[-1], image)
        
        return next_point


def generate_strokes(image, drawn, step_length):
    # generate strokes from the image
    # the step_length is the length of the stroke
    # the image is a 3D tensor with shape (1, height, width)
    # the image should be a gray image
    # the image should be a blured image
    # the return value is a list of points
    # the first point is the maximum point in the image
    # the points
    max_point = get_start_point_of_stroke(image, drawn)
    if max_point is None:
        return []
    points = [max_point]
    while True:
        next_point = get_next_point(points, step_length, image)
        if next_point is None:
            print(f"sort_points_index = {sorted_points_index}, len(points) = {len(points)}")
            if len(points) > 1:
                points.pop(-1)
            break
        if drawn[next_point[0], next_point[1]] == 1:
            break
        points.append(next_point)
    # set all the points to be drawn
    for point in points:
        # set the neighbot points to be drawn
        for x in range(int(point[0] - 10), int(point[0] + 11)):
            for y in range(int(point[1] - 10), int(point[1] + 11)):
                if x < 0 or x >= drawn.shape[0] or y < 0 or y >= drawn.shape[1]:
                    continue
                drawn[point[0], point[1]] = 1
    return points

sorted_points = None
sorted_points_index = 0
def get_start_point_of_stroke(image, drawn):
    # get the start point of the stroke
    # the start point is the maximum point in the image
    # the start point should not be drawn
    global sorted_points, sorted_points_index
    if sorted_points is None:
        sorted_points = list(torch.argsort(image.reshape(-1), descending=True)[:10000])
        # but this time I want to use the shuffled points
        import random
        random.shuffle(sorted_points)
    length = len(sorted_points)
    while sorted_points_index < length:
        point = (int(sorted_points[sorted_points_index] // image.shape[2]), int(sorted_points[sorted_points_index] % image.shape[2]))
        if drawn[point[0], point[1]] == 0:
            return point
        sorted_points_index += 1
    return None


if __name__ == '__main__':
    # image = load_image('modified.jpg')
    # image2 = get_blured_img(image)
    # save_image(image2, 'test2.jpg')
    # image = load_image('modified.jpg')
    # print(image.shape)
    # print(get_next_point([(1, 2), (2, 1), (3, 2), (4, 4), (5, 6)], 2))
    image = load_image("result.jpg")
    drawn = torch.zeros(image.shape[1:])
    import turtle
    turtle.speed(0)
    turtle.pensize(0.1)
    while True:
        points = generate_strokes(image, drawn, 4)
        if len(points) == 0:
            break
        if len(points) < 5:
            continue
        # print(points)
        turtle.penup()
        turtle.goto(int(points[0][1] / 2 - 175), int(150 - points[0][0] / 2))
        turtle.pendown()
        for point in points[::4]:
            turtle.goto(int(point[1] / 2 - 175), int(150 - point[0] / 2))
    input()
    pass
