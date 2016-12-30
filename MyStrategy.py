from model.ActionType import ActionType
from model.MinionType import MinionType
from model.SkillType import SkillType
from model.ProjectileType import ProjectileType
from model.BuildingType import BuildingType
from model.StatusType import StatusType
from model.Faction import Faction
from model.LaneType import LaneType
from model.Game import Game
from model.Move import Move
from model.Wizard import Wizard
from model.World import World
from model.Unit import Unit
from model.CircularUnit import CircularUnit
from model.Message import Message


from math import pi, cos, copysign, sin, acos, asin, sqrt, floor

# Constants
IN_RANGE = 1
OUT_OF_RANGE = 0

WIZARD = 0
MINION = 1
BUILDING = 2
TREE = 3

ACTION   = (1<<0)
ROTATION = (1<<1)
MOVEMENT = (1<<2)

double_pi = 2 * pi

def get_best(unit_list, criteria, initial_value = 0):
  max_value = initial_value
  result = None
  iterator = unit_list
  for u in iterator:
    value = criteria(u)
    if value == 0:
      continue
    if value > max_value:
      result = u
      max_value = value
  return result, max_value
  
def check_los(start: Unit, end: Unit):
  ## Returns True if start and end are on line of sight
  start_edge = get_edge(start)
  end_edge = get_edge(end)
  if start_edge == 0 or end_edge == 0:
    return True
  return bool(get_edge(start) & get_edge(end))

def get_edge(u: Unit, tolerance: int = 400):
  result = 0
  from_right = 4000 - tolerance
  if u.x <= tolerance:
    result = result | (1 << 9)
  if u.y <= tolerance:
    result = result | (1 << 3)
  if u.x >= from_right:
    result = result | (1 << 6)
  if u.y >= from_right:
    result = result | (1 << 12)
  if from_right < u.x + u.y < 4000 + tolerance:
    if u.x <= 2000:
      result = result | (1 << 24)
    if u.x >= 2000:
      result = result | (1 << 18)
  if -tolerance < u.x - u.y < tolerance:
    if u.x <= 2000:
      result = result | (1 << 17)
    if u.x >= 2000:
      result = result | (1 << 20)
  return result


def get_lane(edge: int):
  if edge & ((1<<24) | (1<<18)):
    return LaneType.MIDDLE
  elif edge & ((1<<9) | (1<<3) | (1<<17)):
    return LaneType.TOP
  elif edge & ((1<<12) | (1<<6) | (1<<20)):
    return LaneType.BOTTOM
  else:
    return -1

def get_best_skill(wiz: Wizard, dst_to_target: float):
  skillz = [
    ActionType.MAGIC_MISSILE,
    ActionType.FROST_BOLT,
    ActionType.FIREBALL,
    ]
  manacost = {
    ActionType.MAGIC_MISSILE: 12,
    ActionType.FROST_BOLT: 36,
    ActionType.FIREBALL: 48,
  }
  evaluator = (lambda x: 
    ((3-(x%3)) * x * int(not (x==ActionType.FIREBALL and
    dst_to_target < wiz.radius*3)) * int(wiz.mana >= manacost[x]) *
    int(wiz.remaining_cooldown_ticks_by_action[x] == 0) *
    int(x == ActionType.MAGIC_MISSILE or
        (x*5-6) in wiz.skills)))
  best_spell, spell_value = get_best(skillz, evaluator)
  if spell_value == 0 or best_spell is None:
    best_spell = skillz[0]
  return best_spell, spell_value

def normalize_angle(angle: float):
  while angle < pi:
    angle += double_pi
  while angle > pi:
    angle -= double_pi
  return angle

def angle_between(angle_1: float, angle_2: float):
  max_angle = max(angle_1, angle_2)
  min_angle = min(angle_1, angle_2)
  return normalize_angle(max_angle - min_angle)/2 + min_angle

def near_edge(u: CircularUnit):
  indent = 100
  xcoef = abs(2000 - u.x) - (2000 - u.radius - indent)
  ycoef = abs(2000 - u.y) - (2000 - u.radius - indent)
  if xcoef > 0:
    return xcoef/indent
  if ycoef > 0:
    return ycoef/indent
  return 0

def get_sector_between_angles(fangle: float, sangle: float):
  return abs(normalize_angle(fangle - sangle))

def get_direction_and_sector(triangle: tuple, me: Unit):
  angles = [0, 0, 0]
  for i, vertex in enumerate(triangle):
    angles[i] = me.get_angle_to_unit(vertex)
  max_sector = 0
  for i, angs in enumerate(((1, 2),(0, 2),(0, 1))):
    k, l = angs
    sector = get_sector_between_angles(angles[k], angles[l])
    if sector > max_sector:
      max_sector = sector
      direction = angle_between(angles[k], angles[l])
  return direction, max_sector

class MyStrategy:
  initialized = False
  
  action_to_radius = {
    ActionType.MAGIC_MISSILE: 10,
    ActionType.FROST_BOLT: 15,
    ActionType.FIREBALL: 20,
  }
  
  state = "" # for debug
  chasing_bonus = False # We are stoping other wizards when chasing
  
  forbidden_zones = list() # Its better to avoid theese while pathfind
  last_path = [] # Last path calculated. Just to reduce processing
                 # time
  visited = False # Whether start point is visited?
  rotation_direction = 0 # Direction to keep when bypassing objects
  
  towers_protected = [True, True, True]
  locks = 0
  bonus_speed = 1 # bonus speed factor by talents
  last_speed = 0
  last_x = 0
  last_y = 0 # Necessary for checking if wizard got stuck
  strafe_timer = 0 # changing strafe direction 
  strafe_dir = 1
  dont_dive = False # Dont dive under tower, it is dangerous
  md_skill = 0 # Magic damage bonus skills learned
  last_lane = 1 # Last selected lane. if it pushed much better,
                # the strategy must switch the lane
  learn_order = [
    SkillType.FIREBALL,
    SkillType.FROST_BOLT,
    SkillType.SHIELD,
    SkillType.ADVANCED_MAGIC_MISSILE,
    SkillType.HASTE,
  ]
  talent_tree = [
    [
      SkillType.RANGE_BONUS_PASSIVE_1,
      SkillType.RANGE_BONUS_AURA_1,
      SkillType.RANGE_BONUS_PASSIVE_2,
      SkillType.RANGE_BONUS_AURA_2,
      SkillType.ADVANCED_MAGIC_MISSILE,
    ],
    [
      SkillType.MAGICAL_DAMAGE_BONUS_PASSIVE_1,
      SkillType.MAGICAL_DAMAGE_BONUS_AURA_1,
      SkillType.MAGICAL_DAMAGE_BONUS_PASSIVE_2,
      SkillType.MAGICAL_DAMAGE_BONUS_AURA_2,
      SkillType.FROST_BOLT,
    ],
    [
      SkillType.MAGICAL_DAMAGE_ABSORPTION_PASSIVE_1,
      SkillType.MAGICAL_DAMAGE_ABSORPTION_AURA_1,
      SkillType.MAGICAL_DAMAGE_ABSORPTION_PASSIVE_2,
      SkillType.MAGICAL_DAMAGE_ABSORPTION_AURA_2,
      SkillType.SHIELD,
    ],
    [
      SkillType.MOVEMENT_BONUS_FACTOR_PASSIVE_1,
      SkillType.MOVEMENT_BONUS_FACTOR_AURA_1,
      SkillType.MOVEMENT_BONUS_FACTOR_PASSIVE_2,
      SkillType.MOVEMENT_BONUS_FACTOR_AURA_2,
      SkillType.HASTE,
    ],
    [
      SkillType.STAFF_DAMAGE_BONUS_PASSIVE_1,
      SkillType.STAFF_DAMAGE_BONUS_AURA_1,
      SkillType.STAFF_DAMAGE_BONUS_PASSIVE_2,
      SkillType.STAFF_DAMAGE_BONUS_AURA_2,
      SkillType.FIREBALL,
    ],
  ]
  
  
  def __init__(self):
    self.rules = [
      self.rule_dont_get_stuck,
      self.rule_avoid_tower_damage,
      self.rule_on_low_hp,
      self.rule_avoid_bullets_like_a_neo,
      self.rule_dont_die_alone,
      self.rule_dont_let_them_flee,
      self.rule_collect_bonus,
      self.rule_keep_on_the_lane,
      self.rule_haras,
      self.rule_push_tower,
      self.rule_farm_crips,
      #self.rule_follow_friendly,
      self.rule_choose_lane,
      #lambda: self.print_state("End of chain!")
    ]
    self.forest_triangles = {
      "west": (self.dummy_unit(400, 3200),
               self.dummy_unit(400, 800),
               self.dummy_unit(1600, 2000)),
      "south": (self.dummy_unit(800, 3600),
                self.dummy_unit(2000, 2400),
                self.dummy_unit(3200, 3600)),
      "north": (self.dummy_unit(800, 400),
                self.dummy_unit(2000, 1600),
                self.dummy_unit(3200, 400)),
      "east": (self.dummy_unit(2400, 2000),
               self.dummy_unit(3600, 3200),
               self.dummy_unit(3600, 800)),
    }
    self.forest_circles = [
      self.dummy_unit(900, 2000, 500),
      
      self.dummy_unit(3100, 2000, 500),
      
      self.dummy_unit(2000, 900, 500),
      
      self.dummy_unit(2000, 3100, 500),
      
      self.dummy_unit(750, 2350, 350),
      self.dummy_unit(750, 1650, 350),
      
      self.dummy_unit(3250, 2350, 350),
      self.dummy_unit(3250, 1650, 350),
      
      self.dummy_unit(2350, 750, 350),
      self.dummy_unit(1650, 750, 350),
      
      self.dummy_unit(2350, 3250, 350),
      self.dummy_unit(1650, 3250, 350),
    ]
  
  def initialize(self):
    self.max_distance = self.me.get_distance_to(self.world.width, self.world.height)
    self.oposite_faction = int(self.me.faction == 0)
    self.next_bonus_time = [
       self.game.bonus_appearance_interval_ticks,
       self.game.bonus_appearance_interval_ticks
       ]
    self.vis_range = self.me.vision_range * 1.4
    self.bonuses = [
      self.dummy_unit(1200, 1200, self.game.bonus_radius),
      self.dummy_unit(2800, 2800, self.game.bonus_radius),
    ]
    if self.game.skills_enabled:
      self.rules.insert(4, self.rule_stop_hostile_wizard)
      self.rules.insert(4, self.buff_yourself)
      self.rules.insert(1, self.learn_skillz)
    for b in self.world.buildings:      
      if b.type == BuildingType.FACTION_BASE:
        self.base = b
        break # Base will not change its position, so lets save it
    self.initialized = True
    self.halfdiag = sqrt(self.world.width**2 + self.world.height**2)
    self.bullet_speed = {
      ActionType.MAGIC_MISSILE: self.game.magic_missile_speed,
      ActionType.FROST_BOLT: self.game.frost_bolt_speed,
      ActionType.FIREBALL: self.game.fireball_speed,
    }
    lane_width_x = self.world.width / 10
    lane_width_y = self.world.height / 10
    x_right_edge = self.world.width - lane_width_x
    y_down_edge = self.world.height - lane_width_y
    diag_line = self.halfdiag
    diag_fuzz = (lane_width_x + lane_width_y)/2
    diag_min = diag_line - diag_fuzz
    diag_max = diag_line + diag_fuzz
    self.get_lane = lambda u: (LaneType.TOP
      if (u.x < lane_width_x or u.y < lane_width_y) else 
      (LaneType.BOTTOM
       if u.x > x_right_edge or u.y > y_down_edge else 
       (LaneType.MIDDLE if diag_min < u.x + u.y < diag_max else
        -1)))
    # Pathfinding part
    indent = 200
    full_part = self.world.width - 2 * indent
    full = full_part + indent
    half = (full_part)/2 + indent
    self.vertex_coordinates = (
      self.dummy_unit(indent, indent, 850),
      self.dummy_unit(full, indent, 850),
      self.dummy_unit(full, full, 850),
      self.dummy_unit(indent, full, 850),
      self.dummy_unit(half, half, 400),
    )
    self.vertex_neighbours = (
      (1, 4, 3),
      (0, 4, 2),
      (1, 4, 3),
      (0, 4, 2),
      (0, 1, 2, 3)
    )
    half_hypot = full_part*sqrt(2)/2
    self.edge_distance = {
      ((1 << 0) | (1 << 1)) : full_part, # 0 to 1
      ((1 << 1) | (1 << 2)) : full_part, # 1 to 2
      ((1 << 2) | (1 << 3)) : full_part, # 2 to 3
      ((1 << 0) | (1 << 3)) : full_part, # 3 to 0
      ((1 << 4) | (1 << 0)) : half_hypot, # from center to vertexes
      ((1 << 4) | (1 << 1)) : half_hypot,
      ((1 << 4) | (1 << 2)) : half_hypot,
      ((1 << 4) | (1 << 3)) : half_hypot,
    }
    if self.me.master:
      self.friendlies = []
      for w in self.world.wizards:
        if w.faction == self.me.faction and (not w.me):
          self.friendlies.append(w.id)
      self.friendlies.sort()
      self.master_skillz_tree = [
        [
          SkillType.FROST_BOLT,
          SkillType.FIREBALL,
          SkillType.ADVANCED_MAGIC_MISSILE
        ],
        [
          SkillType.FROST_BOLT,
          SkillType.FIREBALL,
          SkillType.ADVANCED_MAGIC_MISSILE
        ],
        [
          SkillType.FROST_BOLT,
          SkillType.HASTE,
          SkillType.ADVANCED_MAGIC_MISSILE,
        ],
        [
          SkillType.SHIELD,
          SkillType.ADVANCED_MAGIC_MISSILE,
          SkillType.FROST_BOLT,
        ],
      ]
      self.master_lanes = [
        LaneType.TOP,
        LaneType.BOTTOM,
        LaneType.MIDDLE,
        LaneType.MIDDLE,
      ]

  def best_target_for_fireball(self, targets):
    distances = dict()
    maxvalue = 0
    besttarget = None
    best_dst = 0
    for i, target in enumerate(targets):
      value = 0
      for j, neighbour in enumerate(targets):
        if i == j:
          continue
        ij_key = (1<<i)|(1<<j)
        if not (ij_key in distances):
          distances[ij_key] = target[1].get_distance_to_unit(
                                neighbour[1])
        value += (int(distances[ij_key] <=
          self.game.fireball_explosion_min_damage_range) *
          (1 + int(neighbour is Wizard) + int(target is Wizard)) +
          int(distances[ij_key] <=
          self.game.fireball_explosion_max_damage_range))
      if (value > maxvalue and
          target[0] > self.game.fireball_explosion_min_damage_range +
          self.me.radius):
        maxvalue = value
        besttarget = target
    if maxvalue < 1:
      return 0, None
    else:
      return besttarget

  def estimate(self):
    self.units = list()
    for dist in range(0, 2):
      self.units.append(list())
      for faction in range(0, 4):
        self.units[dist].append(list())
        for utype in range(0, 4):
          self.units[dist][faction].append(list())
    # sorting minions and wizards by range and sector
    self.wizards_by_lane = ([0,0],[0,0],[0,0])
    for w in self.world.wizards:
      dst = self.me.get_distance_to_unit(w) 
      self.units[int(dst <= self.vis_range
        )][w.faction][WIZARD].append((dst, w))
      edges = get_edge(w)
      if edges == 0:
        continue
      lane = get_lane(edges)
      self.wizards_by_lane[lane][w.faction] += 1
      if lane >= 0 and w.faction == self.me.faction and dst > 0:
        pushed = w.x - w.y
        if self.lanes_pushed[lane] < pushed:
          self.lanes_pushed[lane] = pushed
          self.lane_avangard[lane] = w
    edges_estimated = 0
    for m in self.world.minions:
      dst = self.me.get_distance_to_unit(m) 
      if (m.faction == Faction.NEUTRAL and
          m.remaining_action_cooldown_ticks > 0):
        self.units[int(dst <= self.vis_range
          )][self.oposite_faction][MINION].append((dst, m))
      else:
        self.units[int(dst <= self.vis_range
          )][m.faction][MINION].append((dst, m))
      edges = get_edge(m)
      if edges == 0:
        continue
      lane = get_lane(edges)
      if (m.faction == self.oposite_faction and
          edges != 0 and
          not (edges_estimated & edges)):
        edges_estimated = edges_estimated | edges
        edge = 0
        while edges != 0:
          if edges & 1:
            self.edge_penalty[edge] += 1
          edges = edges >> 1
          edge += 1
      if lane >= 0 and m.faction == self.me.faction:
        pushed = m.x - m.y
        if self.lanes_pushed[lane] < pushed:
          self.lanes_pushed[lane] = pushed
          self.lane_avangard[lane] = m
    for b in self.world.buildings:
      dst = self.me.get_distance_to_unit(b)
      self.units[int(dst <= self.vis_range
        )][b.faction][BUILDING].append((dst, b))
      edges = get_edge(b)
      if edges == 0:
        continue
      lane = get_lane(edges)
      if lane >= 0 and b.faction == self.me.faction:
        pushed = b.x - b.y
        if self.lanes_pushed[lane] < pushed:
          self.lanes_pushed[lane] = pushed
          self.lane_avangard[lane] = b
    nearest_crip, crip_dst = self.get_nearest(
        self.units[IN_RANGE][self.oposite_faction][MINION] +
        self.units[IN_RANGE][Faction.NEUTRAL][MINION])
    if (crip_dst > 0 and
      crip_dst < self.game.orc_woodcutter_attack_range * 2
      and abs(nearest_crip.angle -
          nearest_crip.get_angle_to_unit(self.me)) < pi/4):
      self.dangerous = True
    else:
      self.dangerous = False
    for lane, pushed in enumerate(self.lanes_pushed):
      if (pushed >= 2300) or (pushed >= 1500 and lane == 1):
        self.towers_protected[lane] = False
    lanes_vertex = [0, 4, 2]
    for i, vertex in enumerate(lanes_vertex):
      if self.lanes_pushed[i] < 0:
        self.vertex_penalty[lanes_vertex[i]] += 2
    self.forest_circles_est = []
    for circle in self.forest_circles:
      self.forest_circles_est.append(
        (self.me.get_distance_to_unit(circle), circle))
  
  def move(self, me: Wizard, world: World, game: Game, move: Move):
    self.act_move = move
    self.world = world
    self.me = me
    self.game = game
    self.locks = 0
    #self.my_player = world.get_my_player()
    self.nearest_tower = None
    self.dangerous = False
    self.dont_dive = False # Dangerous to dive under tower
    self.dive_hard = False # it's possible to approach to the towers
                           # meele range
    self.forbidden_zones = list()
    # Penalties prevents making path through the enemies
    self.edge_penalty = {
      3 : 3,
      6 : 3,
      9 : 1,
      12: 1,
      17: 2,
      18: 3,
      20: 2,
      24: 1,
    }
    self.vertex_penalty = [
      1, 3, 1, 1, 1
    ]
    self.lanes_pushed = [-4000, -4000, -4000]
    self.lane_avangard = [None, None, None]
    if not self.initialized:
      self.initialize()
    self.estimate()
    move.action = ActionType.NONE
    move.turn = 0
    move.speed = 0
    move.strafe_speed = 0
    all_locked = MOVEMENT | ROTATION | ACTION
    
    if self.me.master:
      self.master_move()
    else:
      if not me.messages is None and len(me.messages) > 0:
        lane_to_go = me.messages[0].lane
        if not lane_to_go is None and lane_to_go != self.last_lane:
          self.last_lane = me.messages[0].lane
          self.rule_choose_lane()
        if not me.messages[0].skill_to_learn is None:
          to_learn = me.messages[0].skill_to_learn
          if (not to_learn == self.learn_order[0]) and (not
              to_learn in self.me.skills) and (0 <= to_learn <= 24):
            self.learn_order.insert(0, to_learn)
    for i, rule in enumerate(self.rules):
      rule_status = rule()
      self.print_state(i, "). locks:", self.locks,
        ", rule_status=", rule_status)
      if self.locks == all_locked or rule_status:
        self.print_state("Chain breaked!")
        break
    self.last_speed = (abs(self.act_move.speed) +
       abs(self.act_move.strafe_speed))
  
  def rule_keep_on_the_lane(self):
    if self.locks & MOVEMENT:
      return
    edge = get_edge(self.me)
    avangard = self.lane_avangard[self.last_lane]
    solid = True
    if avangard is None:
      avangard = self.vertex_coordinates[[0, 4, 2][self.last_lane]]
      solid = False
    if edge != 0:
      lane = get_lane(edge)
      if lane != -1:
        dst_to_avangard = self.me.get_distance_to_unit(avangard)
    if edge == 0 or (self.last_lane != lane and
       lane != -1) or dst_to_avangard > 1000:
      self.follow(avangard, solid)
      self.locks = self.locks | MOVEMENT
  
  def rule_collect_bonus(self):
    if abs(self.me.x - self.me.y) > 600 or (self.locks & MOVEMENT):
      return # Wizard should be near mid-line
    edge = get_edge(self.me)
    if edge != 0:
      lane = get_lane(edge)
      if self.wizards_by_lane[lane][self.me.faction] < 2:
        return
    nearest = None
    min_dst = self.max_distance
    to_bonus_appear = [0, 0]
    for i,b in enumerate(self.bonuses):
      if self.next_bonus_time[i] >= self.world.tick_count:
        continue # Last bonus will be unreachable
      to_bonus_appear[i] = (self.next_bonus_time[i] -
                            self.world.tick_index)
      if to_bonus_appear[i] > 1000:
        continue
      wiz, near_wiz_dst = self.get_nearest(self.world.wizards, b)
      if (near_wiz_dst < self.game.wizard_vision_range and
        len(self.world.bonuses) == 0 and
        to_bonus_appear[i] < 0):
        self.next_bonus_time[i] += (
          self.game.bonus_appearance_interval_ticks)
        continue # No bonus here, wait for the next time
      path, dst = self.find_path(self.me, b)
      if (to_bonus_appear[i] <= dst/self.game.wizard_backward_speed):
        if (dst < min_dst and not (dst > near_wiz_dst and
               to_bonus_appear[i] <
               near_wiz_dst/self.game.wizard_forward_speed) and
               (wiz.faction == self.oposite_faction or wiz.me or not
               self.game.raw_messages_enabled)):
          nearest = i
          min_dst = dst
    if nearest is None or min_dst > 2000:
      return
    self.chasing_bonus = True
    self.locks = self.locks | MOVEMENT
    if (to_bonus_appear[nearest] >= 0 and min_dst <=
        self.me.radius + self.bonuses[nearest].radius + 3):
      self.forbidden_zones.append((min_dst, self.bonuses[nearest]))
      self.flee()
      return
    self.follow(self.bonuses[nearest], solid = False)
    self.print_state("Collecting bonus ", self.next_bonus_time)
    return

  def rule_stop_hostile_wizard(self):
    if not self.chasing_bonus:
      return
    if self.locks & (ACTION | ROTATION):
      return
    main_skill = ActionType.FROST_BOLT
    if not SkillType.FROST_BOLT in self.me.skills:
      main_skill = ActionType.MAGIC_MISSILE
      return
    if (self.me.remaining_cooldown_ticks_by_action
      [main_skill] + self.me.remaining_action_cooldown_ticks) > 0:
      return
    enemies = (self.units[IN_RANGE][self.oposite_faction]
      [WIZARD])
    if len(enemies) == 0:
      return
    nearest, dst = self.get_nearest(enemies)
    if dst > self.me.cast_range:
      return
    self.attack_with(nearest, main_skill, dst)
    self.locks = self.locks | ROTATION | ACTION
    self.print_state("Freezing hostile wizard")
    return

  def buff_yourself(self):
    if ((self.locks & ACTION) or
        self.me.remaining_action_cooldown_ticks > 0):
      return
    if (SkillType.SHIELD in self.me.skills and 
        self.me.remaining_cooldown_ticks_by_action[ActionType.SHIELD]
        == 0):
      buff = ActionType.SHIELD
      reqstatus = StatusType.SHIELDED
    elif (SkillType.HASTE in self.me.skills and 
        self.me.remaining_cooldown_ticks_by_action[ActionType.HASTE]
        == 0):
      buff = ActionType.HASTE
      reqstatus = StatusType.HASTENED
    else:
      return
    allies = self.units[IN_RANGE][self.me.faction][WIZARD]
    me_buffed = False
    for dst, wiz in allies:
      if dst > self.me.cast_range:
        continue
      is_buffed = False
      for status in wiz.statuses:
        if status.type == reqstatus:
          is_buffed = True
          if wiz.me:
            me_buffed = True
          break
      if (not is_buffed) and not wiz.me:
        self.act_move.status_target_id = wiz.id
        self.attack_with(wiz, buff)
        self.locks = self.locks | ACTION | ROTATION
        self.print_state("Buffing wizard: ", wiz.id, ", with ",
                         buff)
        return
    if not me_buffed:
      self.act_move.status_target_id = -1
      self.act_move.action = buff
      self.locks = self.locks | ACTION
      self.print_state("Buffing myself with ", buff)
      

  def rule_push_tower(self):
    if (self.locks & (ACTION | ROTATION)):
      return
    if self.nearest_tower is None:
      return
    if (len(self.units[IN_RANGE][self.oposite_faction]
            [BUILDING]) == 0):
      return
    nearest, dst = self.nearest_tower
    if dst > self.me.cast_range:
      if self.dont_dive or (self.locks & MOVEMENT) or self.dangerous:
        return
      self.print_state("Approching to tower")
      self.follow(nearest, waypoints = False)
      self.locks = self.locks | MOVEMENT
    else: 
      if (self.dive_hard and
          dst - nearest.radius >= self.game.staff_range and
          not (self.locks & MOVEMENT)):
        self.print_state("Rushing under tower!")
        self.follow(nearest, waypoints = False)
      best_spell, value = get_best_skill(self.me, dst)
      self.attack_with(nearest, best_spell, dst)
      return True
  
  def print_state(self, *args):
    return
    #result = ""
    #for arg in args:
    #  result += str(arg)
    #if self.state == result:
    #  return
    #self.state = result
    #print("[", self.world.tick_index, "/", self.world.tick_count,
    #  "]:", result)
  
  def rule_avoid_tower_damage(self):
    enemies = self.units[IN_RANGE][self.oposite_faction]
    enemy_towers = enemies[BUILDING]
    allies = self.units[IN_RANGE][self.me.faction]
    ally_wizards = allies[WIZARD]
    if len(enemy_towers) == 0:
      return
    min_dst = self.max_distance
    nearest = None
    self.dont_dive = False
    for dst, tower in enemy_towers:
      #hp_under_tower = self.estimate_hp_under_tower(tower)
      #lowest_combo, value = get_best(ally_wizards,
      #  lambda x: int(x[1].get_distance_to_unit(tower) <=
      #    tower.attack_range) * (-x[1].life/x[1].max_life), -1)
      #if lowest_combo is None:
      #  lowest_dst = 0
      #else:
      #  lowest_dst, lowest = lowest_combo
      lane = get_lane(get_edge(tower))
      is_second = tower.x - tower.y > 2000
      if dst < min_dst and ((not self.towers_protected[lane]) or
        not is_second):
        nearest = tower
        min_dst = dst
        self.nearest_tower = (tower, dst)
      is_shielded = False
      for s in self.me.statuses:
        if s.type == StatusType.SHIELDED:
          is_shielded = True
          break
      double_damage = tower.damage * (2 - int(is_shielded))
      self.print_state("Tower: ", (tower.x, tower.y),
                       #", hp_under=", hp_under_tower,
                       #", lowest_dst=", lowest_dst,
                       ", remaining_cd=",
                       tower.remaining_action_cooldown_ticks,
                       ", dangerous=", self.dangerous,
                       ", t_damage=", tower.damage)
      if ((len(enemies[WIZARD]) + len(enemies[MINION]) > 0 and
          (0.51>self.me.life/self.me.max_life)
          and tower.remaining_action_cooldown_ticks < 100) or
          self.dangerous
          or self.me.life <= tower.damage or (is_second and
          self.towers_protected[lane]) or
          (len(enemies[WIZARD]) + len(enemies[MINION]) == 0
           and self.me.life < double_damage and
           tower.remaining_action_cooldown_ticks <= 50)):
      #if ((tower.remaining_action_cooldown_ticks <= 40 and
        #  (hp_under_tower < double_damage and
        #  self.me.life < double_damage and
        #  len(enemies[WIZARD]) + len(enemies[MINION]) > 0)) or
        #  self.dangerous
        #  or self.me.life <= tower.damage or (is_second and
        #  self.towers_protected[lane])):
        self.forbidden_zones.append((dst, self.dummy_unit(tower.x,
          tower.y, tower.attack_range + self.me.radius)))
        self.dont_dive = True
        if (dst <= (tower.attack_range + self.me.radius) and
           not (self.locks & MOVEMENT)):
        # Flee if no minions under tower
          self.flee_from(tower)
          self.locks = self.locks | MOVEMENT
          self.print_state("Fleeing from tower")
      elif (len(self.units[IN_RANGE][self.oposite_faction][MINION])
             < 2 and (not (nearest is None)) and
             nearest.type != BuildingType.FACTION_BASE):
        #self.print_state("Rushing under tower!")
        self.dive_hard = True
  
  def estimate_hp_under_tower(self, tower: Unit):
    friendly_minions = (self.units[IN_RANGE][self.me.faction]
      [MINION])
    under_tower = filter(
      lambda x: (tower.get_distance_to_unit(x[1]) <
        tower.attack_range), friendly_minions)
    hp_under_tower = 0
    for dst, m in under_tower:
      hp_under_tower += m.life
    return hp_under_tower
    
  def rule_dont_get_stuck(self, force = False):
    last_move = (abs(self.last_x - self.me.x) +
      abs(self.last_y - self.me.y))
    self.last_x = self.me.x
    self.last_y = self.me.y
    if (self.last_speed > 0 and last_move == 0) or force:
      nearest, dst = self.get_nearest(self.world.trees)
      enemies = self.units[IN_RANGE][self.oposite_faction][MINION]
      nearest_crip, crip_dst = self.get_nearest(enemies)
      if 0 < crip_dst <= dst and len(enemies) < 3:
        nearest = nearest_crip
        dst = crip_dst
      if 0 < dst <= (nearest.radius + self.game.staff_range):
        #if self.game.staff_range <= dst + nearest.radius:
        self.print_state("Attacking treep: ", 
          (last_move, self.last_speed))
        self.attack_with(nearest, ActionType.STAFF)
        self.locks = self.locks | ACTION | ROTATION
        return
        #else:
        #  self.print_state("Following treep")
        #  self.follow(nearest)
        #  self.locks = self.locks | MOVEMENT
        #  return
      self.act_move.strafe_speed = (self.get_speed(
        self.game.wizard_strafe_speed)
        * self.get_strafe_dir() * 2)
      self.act_move.turn = pi/4 * self.get_strafe_dir()
      self.act_move.speed = self.get_strafe_dir() * self.get_speed(
        self.game.wizard_forward_speed)
      self.locks = self.locks | MOVEMENT
      self.print_state("Unstucking: ", (last_move, self.last_speed))

  def rule_farm_crips(self):
    minions = self.units[IN_RANGE][self.oposite_faction
      ][MINION]
    if (len(minions) == 0 or (self.locks & (ACTION|ROTATION))):
      return
    best_range = self.me.cast_range
    wizards = self.units[IN_RANGE][self.oposite_faction][WIZARD]
    nearest = None
    if (self.game.skills_enabled and# len(wizards) == 0 and 
      SkillType.FIREBALL in self.me.skills and
      self.me.remaining_cooldown_ticks_by_action[ActionType.FIREBALL]
      == 0):
      best_spell = ActionType.FIREBALL
      dst, nearest = self.best_target_for_fireball(wizards + minions)
    if nearest is None:
      best_spell = ActionType.MAGIC_MISSILE
      best, value = get_best(minions,
        lambda x: ((self.vis_range - x[0])/self.vis_range *
          (1 + int(x[1].type == MinionType.FETISH_BLOWDART) *
          int(not self.dangerous) -
          3 * int(x[0] == 0)) +
          (x[1].max_life - x[1].life)/x[1].max_life *
          int(not self.dangerous)))
      dst, nearest = best
    if nearest is None:
      return
    if ((not self.dangerous) and len(wizards) == 0
        and nearest.type != MinionType.FETISH_BLOWDART
        and (self.nearest_tower is None or
             self.nearest_tower[1]-
             self.nearest_tower[0].attack_range >= dst)):
      best_range = self.game.staff_range
      self.print_state("Smashing crip with stuff")
    if dst > best_range and not (self.locks & MOVEMENT):
      self.print_state("Approaching to crip")
      self.follow(nearest)
      self.locks = self.locks | MOVEMENT
    if dst <= self.me.cast_range:
      self.print_state("Attacking crip with ", best_spell)
      self.attack_with(nearest, best_spell, dst)     
      self.locks = self.locks | ACTION | ROTATION
    #self.print_state("Farming crips")
    

  def rule_follow_friendly(self):
    leaders = dict()
    units = self.units[IN_RANGE][self.me.faction]
    leaders = units[MINION]
    # no leaders near wizard - lets go to far friendly unit
    leader_to_follow, dst = self.get_nearest(leaders)
    if leader_to_follow != None and not (self.locks & MOVEMENT):
      self.follow(leader_to_follow)
      self.locks = self.locks | MOVEMENT
      self.print_state("Following friendly unit")
      return True
  
  def rule_avoid_bullets_like_a_neo(self):
    if self.locks & MOVEMENT:
      return
    for bullet in self.world.projectiles:
      if (bullet.owner_player_id == -1 or
          bullet.faction == self.me.faction):
        continue
      if (bullet.type != ProjectileType.FROST_BOLT and
          self.chasing_bonus):
        continue
      if bullet.type == ProjectileType.FIREBALL:
        explosion = self.game.fireball_explosion_min_damage_range
      else:
        explosion = bullet.radius
      dst_to_center = self.me.get_distance_to_unit(bullet)
      if (self.me.vision_range < dst_to_center): 
        continue
      angle_to_me = bullet.get_angle_to_unit(self.me)
      if abs(angle_to_me) >= pi/2:
        #self.print_state("Bullet " +
        #  str((bullet.x, bullet.y, bullet.angle)) +
        #  " should not be avoided because of angle=" +
        #  str(angle_to_me))
        continue
      dst_to_trajectory = sin(angle_to_me) * dst_to_center
      to_avoid = (self.me.radius + explosion -
                  abs(dst_to_trajectory))
      if to_avoid <= 0:
        #self.print_state("Bullet " +
        #  str((bullet.x, bullet.y, bullet.angle)) +
        #  " should not be avoided because to_avoid=" + str(to_avoid)
        #  + ", angle=" + str(angle_to_me) +
        #  ", dst_to_trj=" + str(dst_to_trajectory) +
        #  ", dst_to_bullet=" + str(dst_to_center))
        continue
      # avoidance
      avoid_angle = (self.me.get_angle_to_unit(bullet) +
        copysign(3*pi/4, -dst_to_trajectory) + self.me.angle)
      safe_place = self.dummy_unit(
        self.me.x + to_avoid * cos(avoid_angle),
        self.me.y + to_avoid * sin(avoid_angle))
      if get_edge(safe_place) == 0 and get_edge(self.me) != 0:
        safe_place = self.dummy_unit(
          self.me.x + to_avoid * cos(-avoid_angle),
          self.me.y + to_avoid * sin(-avoid_angle))
      #self.flee_from(bullet)
      self.follow(safe_place, waypoints = False)
      self.locks = self.locks | MOVEMENT
      self.print_state("Fleeing from bullet")
      #self.print_state("Fleeing from bullet: to_avoid=" +
      #                 str(to_avoid) + ", safe_place = " +
      #                 str((safe_place.x, safe_place.y)) +
      #                 ", me=" + str((self.me.x, self.me.y)) +
      #                 ", dst_to_trj=" + str(dst_to_trajectory) +
      #                 ", avoid_angle=" + str(avoid_angle))
  
  def rule_on_low_hp(self):
    if (self.locks & MOVEMENT):
      return
    life_percent = self.me.life/self.me.max_life
    if (life_percent > 0.4 and
        self.me.remaining_action_cooldown_ticks < 5):
      return
    elif life_percent <= 0.4:
      self.dangerous = True
    enemy_wizards = self.units[IN_RANGE][self.oposite_faction
      ][WIZARD]
    enemy_minions = self.units[IN_RANGE][self.oposite_faction
      ][MINION]
    if len(enemy_wizards) > 0:
      # Keep distance with enemy wizards
      nearest_wiz, wiz_dst = self.get_nearest(enemy_wizards)  
      if nearest_wiz is None:
        return
      self.forbidden_zones.append((wiz_dst, self.dummy_unit(
        nearest_wiz.x, nearest_wiz.y, nearest_wiz.cast_range +
        self.me.radius)))
      if (wiz_dst < nearest_wiz.cast_range + self.me.radius):
        self.flee_from(nearest_wiz)
        self.locks = self.locks | MOVEMENT
        self.print_state("Fleeing from wizards attack_range")
        return
    if (len(enemy_minions) > 0 and
        life_percent <= 0.4):
      nearest = None
      min_dist = self.max_distance
      for dst, m in enemy_minions:
        if (m.type == MinionType.FETISH_BLOWDART and
          dst <= self.game.fetish_blowdart_attack_range * 1.2
          and dst < min_dist):
          min_dist = dst
          nearest = m
        if (dst <= self.game.orc_woodcutter_attack_range * 5 and
          dst <= min_dist):
          nearest = m
          min_dist = dst
      if not nearest is None:
        self.flee_from(nearest)
        self.locks = self.locks | MOVEMENT
        self.print_state("Fleeing from minion on low HP")
      
  def rule_dont_die_alone(self):
    if self.locks & MOVEMENT:
      return
    enemies = (int(self.dangerous) * self.game.wizard_vision_range +
               int(self.me.x-self.me.y > 2500) *
               self.game.wizard_vision_range)
    distance_factor = 0
    friends = 0
    my_dist = self.me.get_distance_to_unit(self.base)
    for dst, e in (self.units[IN_RANGE]
      [self.oposite_faction][MINION]):
      enemies += self.vis_range - dst
    for dst, e in (self.units[IN_RANGE]
      [self.oposite_faction][WIZARD]):
      enemies += self.vis_range - 2 * dst
    for dst, e in (self.units[IN_RANGE]
      [self.me.faction][MINION]):
      #friends += self.vis_range - 2 * dst
      addition = e.get_distance_to_unit(self.base) - my_dist
      if addition > 0:
        friends += addition
    for dst, e in (self.units[IN_RANGE]
      [self.me.faction][WIZARD]):
      val = self.vis_range - 3 * dst
      if val > 0:
        friends += val
    visible_units = enemies + friends  
    if visible_units <= 0:
      return
    self.coeficient = (enemies - friends)/visible_units  
    if self.coeficient >= 0.2:
      self.dangerous = True
    dst_to_base = (my_dist -
                   self.base.radius - self.me.radius)
    if self.coeficient >= 0.3 and dst_to_base > 100:
      self.flee()
      self.locks = self.locks | MOVEMENT
      #self.print_state("Regrouping: " + str((enemies, friends)))

  def rule_choose_lane(self):
    min_pushed = 1
    min_pushed_val = 4000
    push_point = 0
    for lane, pushed in enumerate(self.lanes_pushed):
      if min_pushed_val > pushed:
        min_pushed_val = pushed
        min_pushed = lane
    if (self.locks & MOVEMENT) and (min_pushed_val > -2500):
      return
    #if self.last_lane == -1:
    #  self.last_lane = min_pushed
    if (self.lanes_pushed[self.last_lane] - min_pushed_val < 1000
        or self.world.tick_index < 1000 or
        self.world.tick_index > 18000 or min_pushed_val > -2000 or
        self.wizards_by_lane[self.last_lane][self.me.faction] < 2):
      min_pushed = self.last_lane # Prevent often lane switching
    self.print_state("Choisen lane: ", min_pushed, ", push values:",
                     self.lanes_pushed)
    self.last_lane = min_pushed
    self.locks = self.locks | MOVEMENT
    avangard = self.lane_avangard[min_pushed]
    solid = True
    if avangard == None:
      avangard = self.base
    if self.world.tick_index < 1600:
      avangard = self.vertex_coordinates[(0, 4, 2)[min_pushed]]
      solid = False
    self.follow(avangard, solid)
        
  
  def rule_dont_let_them_flee(self):
    wizards = self.units[IN_RANGE][self.oposite_faction][WIZARD]
    life_percent = self.me.life/self.me.max_life
    if life_percent < 0.5:
      return
    bold_wiz, bold_coef = self.get_nearest(self.world.wizards,
                                           self.base)
    if bold_wiz.faction != self.me.faction and bold_coef < 400:
      self.follow(bold_wiz)
      self.locks = self.locks | MOVEMENT
      return
    if len(wizards) == 0 or (self.locks&(ACTION|ROTATION|MOVEMENT)):
      return
    #has_empower = 0
    #for s in self.me.statuses:
    #  if s.type == StatusType.EMPOWERED:
    #    has_empower = 1
    #missile_damage = (self.game.magic_missile_direct_damage +
    #  (1 + has_empower * self.game.empowered_damage_factor +
    #  self.md_skill*self.game.magical_damage_bonus_per_skill_level))
    for dst, wizard in wizards:
      addition = 0
      for s in wizard.statuses:
        if s.type == StatusType.FROZEN:
          addition += 0.2
        elif s.type == StatusType.HASTENED:
          addition -= 0.1
        elif s.type == StatusType.SHIELDED:
          addition -= 0.1
        elif s.type == StatusType.BURNING:
          addition += 0.05
      if (wizard.life/wizard.max_life < (0.3 + addition) and
          self.coeficient < 0.7):
        if (dst <= self.me.cast_range):
          best_spell, value = get_best_skill(self.me, dst)
          self.attack_with(wizard, best_spell, dst)
          return True
        else:
          self.follow(wizard)
          return True
        

  def rule_haras(self):
    if (self.locks & (ACTION|ROTATION)):
      return
    nearest_minion, minion_dst = self.get_nearest(
      self.units[IN_RANGE][self.oposite_faction][MINION])
    if (minion_dst < self.game.staff_range and
        not nearest_minion is None):
      return
    enemies = self.units[IN_RANGE][self.oposite_faction
      ][WIZARD]
    if len(enemies) == 0:
      return
    best_and_dst, value = get_best(enemies,
      lambda x: (self.vis_range - x[0])/
      self.vis_range + (x[1].max_life - x[1].life)/
      x[1].max_life)
    best_dst, best = best_and_dst
    if best_and_dst == None:
      return
    nearest, min_dst = self.get_nearest(enemies)
    best_spell = 0
    spell_value = 0
    n_frozen = 0
    e_frozen = 0
    dangerous_skill_near = ActionType.MAGIC_MISSILE
    dangerous_skill_best = ActionType.MAGIC_MISSILE
    if self.game.skills_enabled:
      dangerous_skill_near, dsn_value = get_best_skill(nearest,
                                                       min_dst)
      dangerous_skill_best, dsb_value = get_best_skill(best,
                                                       best_dst)
      best_spell, spell_value = get_best_skill(self.me, best_dst)
      for st in best.statuses:
        if st.type == StatusType.FROZEN:
          e_frozen = st.remaining_duration_ticks
          break
      for st in nearest.statuses:
        if st.type == StatusType.FROZEN:
          n_frozen = st.remaining_duration_ticks
          break
    else:
      best_spell = ActionType.MAGIC_MISSILE
      spell_value = int(
        self.me.remaining_cooldown_ticks_by_action[best_spell] == 0)
    if best_spell == ActionType.FIREBALL:
      best_dst -= self.game.fireball_explosion_max_damage_range
    enemy_cds = max(best.remaining_action_cooldown_ticks,
      best.remaining_cooldown_ticks_by_action[dangerous_skill_best],
      e_frozen, best.get_angle_to_unit(self.me)/
      self.game.wizard_max_turn_angle)
    nearest_cds = max(nearest.remaining_action_cooldown_ticks,
      nearest.remaining_cooldown_ticks_by_action
        [dangerous_skill_near],
      n_frozen, nearest.get_angle_to_unit(self.me)/
      self.game.wizard_max_turn_angle)
    if (self.me.remaining_action_cooldown_ticks == 0 and
        spell_value > 0 and
        (min(nearest_cds, enemy_cds) > 5 or 
         value >= 1)):
      if (best_dst > self.me.cast_range and
          not (self.locks & MOVEMENT)):
        #self.print_state("Approaching to hit wizard: " + str(best))
        self.follow(best)
        self.locks = self.locks | MOVEMENT
      elif best_dst <= self.me.cast_range:
        self.print_state("Hitting and fleeing")
        self.attack_with(best, best_spell, best_dst)
        return True
        #self.backpedal_from(nearest)
    elif (self.me.remaining_action_cooldown_ticks == 0 and
          spell_value > 0 and nearest_cds > 5):
      if best_spell == ActionType.FIREBALL:
        min_dst -= self.game.fireball_explosion_max_damage_range
      if (min_dst > self.me.cast_range and
          not (self.locks & MOVEMENT)):
        #self.print_state("Approaching to hit wizard: " + str(best))
        self.follow(nearest, waypoints = False)
        self.locks = self.locks | MOVEMENT
      elif min_dst <= self.me.cast_range:
        self.print_state("Hitting and fleeing")
        self.attack_with(nearest, best_spell, min_dst)
        return True
    elif (min_dst <= nearest.cast_range + self.me.radius + 5
          and not (self.locks & MOVEMENT)):
      self.print_state("Getting out of wizard attack range")
      #self.strafe_speed = self.get_speed(
      #  self.game.wizard_strafe_speed *
      #  self.get_strafe_dir())
      self.flee_from(nearest)
      self.locks = self.locks | MOVEMENT
  
  def master_move(self):
    skill_message = [None, None, None, None]
    wizards = [None, None, None, None]
    if self.world.tick_index > 2000:
      need_reinforcement = -1
      has_reinforcement = -1
      for i, lane in enumerate(self.wizards_by_lane):
        advantage = lane[self.me.faction]-lane[self.oposite_faction]
        if advantage < -1 or lane[self.me.faction] == 0:
          need_reinforcement = i
        if advantage > 0 and lane[self.me.faction] > 1:
          has_reinforcement = i
      if need_reinforcement >= 0 and has_reinforcement >= 0:
        to_switch = -1
        second = False
        for i, lane in enumerate(self.master_lanes):
          if lane == has_reinforcement:
            if second:
              to_switch = i
              break
            else:
              second = True
        if to_switch >= 0:
          self.master_lanes[to_switch] = need_reinforcement
    for w in self.world.wizards:
      if w.id in self.friendlies:
        index = self.friendlies.index(w.id)
        wizards[index] = w
    for i, tree in enumerate(self.master_skillz_tree):
      if wizards[i] is None:
        continue
      for skill in tree:
        if not skill in wizards[i].skills:
          skill_message[i] = skill
          break
    self.act_move.messages = [None, None, None, None]
    for i, lane in enumerate(self.master_lanes):
      if wizards[i] is None:
        continue
      self.act_move.messages[i] = (
        Message(lane, skill_message[i], None))
  
  def learn_skillz(self):
    s = self.learn_order[0]
    if not s in self.me.skills:
      for branch in self.talent_tree:
        if s in branch:
          for skill in branch:
            if not skill in self.me.skills:
              self.act_move.skill_to_learn = skill
              return
    else:
      if (s == SkillType.MOVEMENT_BONUS_FACTOR_PASSIVE_1 or
          s == SkillType.MOVEMENT_BONUS_FACTOR_PASSIVE_2):
        self.bonus_speed *= (
          self.game.movement_bonus_factor_per_skill_level)
      elif (s == SkillType.MAGICAL_DAMAGE_BONUS_PASSIVE_1 or
            s == SkillType.MAGICAL_DAMAGE_BONUS_PASSIVE_2):
        self.md_skill += 1
      self.learn_order.remove(s)
      self.print_state("Learned ", s)
  
  
  def get_nearest(self, unit_list, to_unit = None):
    if len(unit_list) == 0:
      return None, 0
    if to_unit == None:
      to_unit = self.me
    ruler = lambda x: (-to_unit.get_distance_to_unit(x) - self.max_distance * int(x == to_unit))
    if type(unit_list[0]) is tuple:
      ruler = lambda x: -x[0] - self.max_distance * int(x[0] == 0)
    u, dst = get_best(unit_list, ruler,
                           -self.max_distance)
    if u == None:
      return None, 0
    if type(u) is tuple:
      u = u[1]
    return u, -dst
    
  def attack_with(self, enemy: Unit, action: int,
                  enemy_dst: float = 0):
    if enemy_dst == 0:
      enemy_dst = self.me.get_distance_to_unit(enemy)
    if action in self.bullet_speed:
      bullet_speed = self.bullet_speed[action]
      enemy_in_future = self.dummy_unit(
        enemy.x + enemy.speed_x*enemy_dst/bullet_speed/2,
        enemy.y + enemy.speed_y*enemy_dst/bullet_speed/2,
        enemy.radius)
      self.act_move.turn =self.me.get_angle_to_unit(enemy_in_future)
    else:
      self.act_move.turn =self.me.get_angle_to_unit(enemy)
    staff_cd = (self.me.remaining_cooldown_ticks_by_action
                [ActionType.STAFF])
    act_cd = self.me.remaining_cooldown_ticks_by_action[action]
    if (action == ActionType.STAFF and
       staff_cd > self.me.remaining_cooldown_ticks_by_action[
        ActionType.MAGIC_MISSILE]):
        action = ActionType.MAGIC_MISSILE
    elif (enemy_dst - enemy.radius <= self.game.staff_range and
        staff_cd == 0 and (act_cd > 0 or self.me.mana < 12) and
        action != ActionType.SHIELD):
      action = ActionType.STAFF
    if abs(self.act_move.turn) < self.game.staff_sector/4:
      self.act_move.action = action
      self.act_move.cast_angle = self.act_move.turn
      self.act_move.min_cast_distance = enemy_dst - enemy.radius
      self.act_move.max_cast_distance = enemy_dst + enemy.radius

  def get_wall_potencial(self):
    field_range = self.me.radius * 3
    px, py = (0,0)
    if self.me.x < field_range:
      px += field_range - self.me.x
    elif self.me.x > self.world.width - field_range:
      px += self.world.width - field_range - self.me.x
    if self.me.y < field_range:
      py += field_range - self.me.y
    elif self.me.y > self.world.height - field_range:
      py += self.world.height - field_range - self.me.y
    return (px, py)

  def get_forest_potencial(self):
    hardedge = get_edge(self.me, tolerance = 300)
    if hardedge != 0:
      return (0,0)
    wayout, dst = self.from_forest()
    if dst == 0:
      self.print_state("WTF???")
      return (0,0)
    decreaser = dst/1000
    if decreaser > 1:
      decreaser = 1
    px = (wayout.x - self.me.x)*decreaser/dst
    py = (wayout.y - self.me.y)*decreaser/dst
    #for direction, triangle in self.forest_triangles.items():
    #  for vertex in triangle:
    #    v_dst = self.me.get_distance_to_unit(vertex)
    #    if v_dst < 200:
    #      pvx, pvy = self.get_potencial(vertex)
    #      px += pvx/2
    #      py += pvy/2
    return (px, py)

  def from_forest(self):    
    diag_val = (self.me.x + self.me.y)/2
    diag_pnt = self.dummy_unit(diag_val, diag_val)
    diag_dst = self.me.get_distance_to_unit(diag_pnt)
    halfdiag = (self.world.width + self.world.height) / 2
    midlane_x = halfdiag - self.me.y + self.me.x
    midlane_y = halfdiag - midlane_x
    midlane_pnt = self.dummy_unit(midlane_x, midlane_y)
    midlane_dst = self.me.get_distance_to_unit(midlane_pnt)
    left_dst = self.me.x - 200
    left_pnt = self.dummy_unit(200, self.me.y)
    right_dst = self.world.width - 200 - self.me.x
    right_pnt = self.dummy_unit(self.world.width - 200, self.me.y)
    top_dst = self.me.y - 200
    top_pnt = self.dummy_unit(self.me.x, 200)
    bottom_dst = self.world.height - 200 - self.me.y
    bottom_pnt = self.dummy_unit(self.me.x,
                                 self.world.height - 200)
    ways = [
      (midlane_dst, midlane_pnt),
      (diag_dst, diag_pnt),
      (left_dst, left_pnt),
      (right_dst, right_pnt),
      (top_dst, top_pnt),
      (bottom_dst, bottom_pnt),
    ]
    self.print_state("Get outta the forest: ways=",
                       ways)
    return self.get_nearest(ways)

  def get_potencial(self, obstacle: Unit, distance = None):
    if distance is None:
      distance = self.me.get_distance_to_unit(obstacle)
    sum_radius = self.me.radius + obstacle.radius
    distance = distance - sum_radius
    amplifier = 1
    if distance <= 0:
      distance = 0.0000000001
      amplifier = 10
    #field_range = sum_radius + self.me.radius * 2
    #if distance < field_range:
    #  amplifier = 1
    ##elif distance > 2*field_range:
    ##  return (0, 0)
    #else:
    #  amplifier = 1
    #angle = obstacle.get_angle_to_unit(self.me) - obstacle.angle
    #x_rad = sum_radius * cos(angle)
    #y_rad = sum_radius * sin(angle)
    px = (self.me.x - obstacle.x)*amplifier/(distance*distance)
    py = (self.me.y - obstacle.y)*amplifier/(distance*distance)
    return (px, py)

  def sum_potencials(self, target: Unit, potencials: list,
                     t_dst: float, second: Unit = None):
    sum_x = 0
    sum_y = 0
    overlap = 0
    cutoff = 0.001
    for x, y in potencials:
      if abs(x) > cutoff or abs(y) > cutoff:
        sum_x += x
        sum_y += y
        overlap += 1
    if t_dst <= 0:
      t_dst = 1
    x = sum_x * overlap + (target.x - self.me.x)/t_dst
    y = sum_y * overlap + (target.y - self.me.y)/t_dst
    if not second is None:
      dst = self.me.get_distance_to_unit(second)
      if dst <= 0:
        dst = 1
      x += (second.x - self.me.x)/dst
      y += (second.y - self.me.y)/dst
    self.print_state("Resulting potencial: (", x, ", ", y,
                     "), overlap=", overlap)
    if x == 0 and y == 0:
      # Local minima! that sucks
      self.print_state("Local minima encountered!!!")
      self.rule_dont_get_stuck()
      return
    return self.dummy_unit(0, 0).get_angle_to(x, y)

  def find_direction(self, target: Unit, dst_to_follow: float = 0,
                     second: Unit = None):
    if dst_to_follow == 0:
      dst_to_follow = self.me.get_distance_to_unit(target)
    objects = (self.units[IN_RANGE][self.me.faction] +
      self.units[IN_RANGE][Faction.NEUTRAL] + self.units
      [IN_RANGE][self.oposite_faction])
    objects += [self.forest_circles_est]
    objects += [self.forbidden_zones]
    potencials = [
      self.get_wall_potencial(),
      #self.get_forest_potencial(),
    ]
    for kind, objects_list in enumerate(objects):  
      is_movable = (kind == MINION) or (kind == WIZARD)
      for dst_to_center, theobj in objects_list:
        obj = None
        if is_movable:
          obj = self.dummy_unit(theobj.x + theobj.speed_x,
                                theobj.y + theobj.speed_y,
                                theobj.radius)
        else:
          obj = theobj
        if dst_to_center > dst_to_follow or dst_to_center == 0:
          continue
        potencials.append(self.get_potencial(obj, dst_to_center))
    self.print_state("Potencials: ", potencials)
    result = self.sum_potencials(target, potencials,
                                 dst_to_follow, second)
    if result is None:
      return None
    return normalize_angle(result - self.me.angle)
    



  def follow(self, origin_target: CircularUnit, waypoints = True,
             solid = True):
    target = origin_target
    dist = self.me.get_distance_to_unit(origin_target)
    second_target = None
    if waypoints and not check_los(self.me, origin_target):
      if (len(self.last_path) == 0 or not
          (check_los(self.me,
                     self.vertex_coordinates[self.last_path[0]]) and
           check_los(target,
                     self.vertex_coordinates[self.last_path[-1]]))):
        self.last_path, total_dst = self.find_path(self.me, target)
        self.visited = False
      if len(self.last_path) > 0:
        target = self.vertex_coordinates[self.last_path[0]]
      dist = self.me.get_distance_to_unit(target)
      if (self.visited == False and len(self.last_path) > 0 and
          dist <= self.vertex_coordinates[self.last_path[0]].radius):
        self.visited = True
      if self.visited:
        second_target = None
        if len(self.last_path) > 1:
          second_target = self.vertex_coordinates[self.last_path[1]]
        else:
          second_target = origin_target
        #angle_to_sum = self.me.get_angle_to_unit(second_target)
        if check_los(self.me, second_target):
          self.last_path.pop(0)
          target = second_target
          second_target = None
          self.visited = False
        dist = self.me.get_distance_to_unit(target)
    else:
      self.visited = False
      self.last_path = []
      dist = self.me.get_distance_to_unit(target)
    if target != origin_target:
      solid = False
    #angle = self.me.get_angle_to_unit(target)
    #if angle_to_sum != 0:
    #  self.print_state("Angle (before, sum): " +
    #    str((angle,angle_to_sum)))
    #  angle = angle_between(angle, angle_to_sum)
    self.print_state("Following: " + str((target.x, target.y, dist,
      self.last_path, self.me.angle,
      origin_target.x, origin_target.y, solid)))
    #angle = self.check_on_the_way(angle,
    #                              dist - int(solid) * target.radius)
    angle = self.find_direction(target, dist - int(solid) * target.radius,
                                second_target)
    self.print_state("After correction: " + str(angle))
    if angle == None:
      # Angle can not be corrected, we stuck between units
      return
    #from_edge = self.check_on_edge(self.me, self.me.radius * 2)
    #if not from_edge is None:
    #  angle = angle_between(angle, from_edge)
    if not (self.locks & ROTATION):
      self.act_move.turn = angle
    cos_angle = cos(angle)
    directed_speed = self.get_speed(self.game.wizard_forward_speed)
    if cos_angle < 0:
      directed_speed = self.get_speed(
        self.game.wizard_backward_speed)
    self.act_move.speed = directed_speed * cos_angle
    self.act_move.strafe_speed =  self.get_speed(
      self.game.wizard_strafe_speed) * sin(angle)
    #self.print_state("Summary: ",
    #  sqrt((self.act_move.strafe_speed/3.0)**2 +
    #       (self.act_move.speed/4.0)**2))
    #self.print_state("Summary back: ",
    #  sqrt((self.act_move.strafe_speed/3.0)**2 +
    #       (self.act_move.speed/3.0)**2))

  def follow_xy(self, x: float, y: float, radius: float = 0):
    target = self.dummy_unit(x, y, radius)
    self.follow(target)
  
  def flee(self):
    to_base = self.me.get_distance_to_unit(self.base)
    best, value = get_best(self.units[OUT_OF_RANGE]
      [self.me.faction][BUILDING],
      lambda x: ((to_base - x[0])/to_base) +
        x[1].life/x[1].max_life)
    if best is None or best[0] < 200:
      if to_base > 200:
        best = (0, self.base)
      else:
        best = (0, self.dummy_unit(0, self.world.height, 10))
    dst, nearest = best
    self.follow(nearest)
    return self.me.get_angle_to_unit(nearest)
  
  def backpedal_from(self, target: Unit):
    self.follow(target, waypoints = False)
    self.act_move.speed = -self.act_move.speed
  
  def get_strafe_dir(self):
    self.strafe_timer += 2
    if self.strafe_timer > 600:
      self.strafe_timer = 0
      self.strafe_dir *= -1
    return self.strafe_dir
  
  def flee_from(self, target_from: Unit):
    #on_edge = (self.world.width - diameter < self.me.x or
    #  self.me.x < diameter or self.me.y < diameter or
    #  self.me.y > self.world.height - diameter)
    #self.flee()
      #strafe_angle = self.act_move.turn
    
    #print((self.act_move.turn, self.me.angle))
    #return
    if get_edge(self.me) == 0:
      self.follow(self.base)
      return
    #fleeangle = self.flee()
    angle = (self.me.get_angle_to_unit(target_from) - pi)
    angle = normalize_angle(angle)
    #angle = angle_between(angle, fleeangle)
    #near_coef = near_edge(self.me)
    #angle = normalize_angle(near_coef * fleeangle +
    #                             (1-near_coef)*angle)
    target = self.dummy_unit(self.me.x + 300 * 
      cos(angle + self.me.angle),
      self.me.y + 300 * sin(angle + self.me.angle))
    #self.print_state("Flee target: " + str((target.x, target.y,
    #  angle, self.me.angle, self.me.x, self.me.y)))
    if get_edge(target) != 0:
      self.follow(target, solid = False)
    else:
      self.flee()

  def get_speed(self, base_speed):
    for status in self.me.statuses:
      if status.type == StatusType.HASTENED:
        return base_speed * (1 +
          self.game.hastened_movement_bonus_factor)
    return base_speed * self.bonus_speed


  def dummy_unit(self, x, y, radius = 0):
    return CircularUnit(0, x, y, 0, 0, 0, None, radius)
  
  def find_path(self, start: Unit, end: Unit):
    ## Find the route using modified Lee's algorithm
    ## Its pretty easy and suitable for our little graph
    ## Returns points to visit and distance of found path
    # check if points in LoS
    if check_los(start, end):
      # if start and end in los - make straight path
      return [], start.get_distance_to_unit(end)
    vertex_index = [0, 0, 0, 0, 0]
    last_wave = []
    last_index = []
    next_wave = 0
    # lets check which points of our graph in los with start
    # and end
    for vertex, point in enumerate(self.vertex_coordinates):
      if check_los(start, point):
        vertex_index[vertex] = (start.get_distance_to_unit(point) *
                                self.vertex_penalty[vertex])
        next_wave = next_wave | (1 << vertex)
      if check_los(end, point):
        last_wave.append(vertex)
        last_index.append(end.get_distance_to_unit(point) * 
                          self.vertex_penalty[vertex])
    if next_wave == 0: # whoops we are in forest
      min_dst = -1
      nearest = -1
      for vertex, point in enumerate(self.vertex_coordinates):
        dst = start.get_distance_to_unit(point)
        if min_dst > dst or min_dst == -1:
          min_dst = dst
          next_wave = 1 << vertex
          nearest = vertex
      vertex_index[nearest] = min_dst
    #self.print_state("Find path: " + str((vertex_index, start.x,
    #                 start.y, end.x, end.y)))
    if len(last_wave) == 0: # WHAT??? Target in the forest?!
      min_dst = -1
      nearest = -1
      for vertex, point in enumerate(self.vertex_coordinates):
        dst = end.get_distance_to_unit(point)
        if min_dst > dst or min_dst == -1:
          min_dst = dst
          nearest = vertex
      last_wave = [nearest]
      last_index = [min_dst]
    first_wave = next_wave
    # Performing search
    while not next_wave == 0:
      wave = next_wave
      next_wave = 0
      vertex = 0
      while wave != 0 and vertex < 5:
        if (1 & wave):
          parent_index = vertex_index[vertex]
          for neighbour in self.vertex_neighbours[vertex]:
            edge = (1<<vertex)|(1<<neighbour)
            distance = self.edge_distance[edge]
            penalty = self.edge_penalty[edge]
            v_penalty = self.vertex_penalty[neighbour]
            summary_index = (distance * penalty * v_penalty
                              + parent_index)
            if (vertex_index[neighbour] == 0 or
                vertex_index[neighbour] > summary_index):
              vertex_index[neighbour] = summary_index
              next_wave = next_wave | (1 << neighbour)
        wave = wave >> 1
        vertex += 1
    # Perform the last wave
    min_i = -1
    min_index = -1
    for i, neighbour in enumerate(last_wave):
      index = last_index[i] + vertex_index[neighbour]
      #self.print_state("Index: " + str(index) + ", last_index="
      #  + str(last_index[i]) + ", last_wave=" + str(neighbour) +
      #  ", vertex_index=" + str(vertex_index[neighbour]))
      if (index < min_index or min_index == -1):
        min_i = i
        min_index = index
    end_point = last_wave[min_i]
    total_distance = (last_index[min_i] /
                      self.vertex_penalty[end_point])
    # Restoring path
    new_path = []
    next_wave = end_point
    #self.print_state("First wave: " + str(first_wave) +
    #  ", next_wave=" + str(1 << next_wave))
    while next_wave != None:
      new_path.insert(0, next_wave)
      prev_wave = next_wave
      next_wave, value = get_best(
        self.vertex_neighbours[next_wave],
        lambda x: -vertex_index[x], -vertex_index[prev_wave])
      if (next_wave is None or ((1<<prev_wave) & first_wave)):
        total_distance += (vertex_index[prev_wave]/
                           self.vertex_penalty[prev_wave])
        break
      else:
        edge = (1<<next_wave)|(1<<prev_wave)
        total_distance += self.edge_distance[edge]
    #self.print_state("Found path: " + str((new_path,
    #       total_distance, vertex_index, last_wave,
    #       last_index)))
    return new_path, total_distance

#  def check_on_edge(self, target: CircularUnit, sum_radius: float):
#    ## Returns direction outta the wall or None if object is
#    ## far from wall
#    result = None
#    if target.x - sum_radius < 0:
#      result = -self.me.angle
#    elif target.x + sum_radius > self.world.width:
#      result = normalize_angle(pi - self.me.angle)
#    y_result = None
#    if target.y - sum_radius < 0:
#      y_result = normalize_angle(pi/2 - self.me.angle)
#    elif target.y + sum_radius > self.world.height:
#      y_result = normalize_angle(-pi/2 - self.me.angle)
#    if not (y_result is None):
#      if result is None:
#        result = y_result
#      else:
#        result = angle_between(result, y_result)
#    return result
#
#  def check_on_the_way(self, angle: float, dst_to_follow: int = 0,
#    ignore_forbidden: bool = False):
#    ## Returns angle wizard to be rotated to avoid collizion
#    ## with friendly unit.
#    ## if target_dst provided - returs strafe distance instead
#    if dst_to_follow == 0:
#      dst_to_follow = self.max_distance
#    diameter = self.me.radius * 2
#    objects = (self.units[IN_RANGE][self.me.faction] +
#      self.units[IN_RANGE][Faction.NEUTRAL] + self.units
#      [IN_RANGE][self.oposite_faction])
#    if not ignore_forbidden:
#      objects += [self.forest_circles_est]
#      objects += [self.forbidden_zones]
#    objects_around = []
#    #print("Summary:", objects)
#    for kind, objects_of_type in enumerate(objects):  
#      is_movable = (kind == MINION) or (kind == WIZARD)
#      #print("oot:", (kind, objects_of_type))
#      for dst_to_center, theobj in objects_of_type:
#        obj = None
#        if is_movable:
#          obj = self.dummy_unit(theobj.x + theobj.speed_x,
#                                theobj.y + theobj.speed_y,
#                                theobj.radius)
#        else:
#          obj = theobj
#        dst_to_edge = dst_to_center - obj.radius
#        if dst_to_center == 0 or dst_to_edge >= dst_to_follow:
#          #self.print_state("Skipping object " +
#          #  str((obj.x, obj.y)) + ", to_center=" +
#          #  str(dst_to_center) + ", to_edge=" +
#          #  str(dst_to_edge) + ", to_follow" +
#          #  str(dst_to_follow)) 
#          continue
#        sum_radius = obj.radius + self.me.radius + int(
#          not ignore_forbidden)
#        ang_to_obj = self.me.get_angle_to_unit(obj)
#        sin_halfsector = sum_radius/dst_to_center
#        if abs(sin_halfsector) > 1:
#          result = normalize_angle(ang_to_obj + pi)
#          result = self.check_on_the_way(result,
#            sum_radius - dst_to_center + self.me.radius,
#            ignore_forbidden = True)
#          #self.print_state("Fleeing from forbidden zone: " +
#          #  str((sum_radius, dst_to_center, result,
#          #  angle)))
#          return result
#        halfsector = asin(sin_halfsector)
#        # угол между прямой к центру юнита и
#        # касательной к окружности вокруг юнита с радиусом
#        # равным сумме радиуса юнита и self.me.radius
#        objects_around.append(
#          (obj, dst_to_edge, ang_to_obj, halfsector, sum_radius))
#    #for name, triangle in self.forest_triangles.items():
#    #  direction, sector = get_direction_and_sector(triangle,
#    #                                               self.me)
#    #  nearest_pt, dst = self.get_nearest(triangle)
#    #  objects_around.append(
#    #    (nearest_pt, dst, direction, sector/2, self.me.radius))
#    shift = 0
#    d = dst_to_follow
#    last_angle = angle
#    while 0 <= shift < pi:
#      new_angle, d = self.find_bypass(objects_around,
#                                         last_angle, d, 1)
#      shift_step = normalize_angle(new_angle - last_angle)
#      shift += shift_step
#      self.print_state("Positive: shift=", shift, ", new_angle=",
#                       new_angle, ", last_angle=", last_angle,
#                       ", d=", d)
#      if shift_step < 0:
#        self.print_state("Bypass in positive is not found")
#        shift = pi + 1
#        break
#      elif shift_step == 0:# or (last_d != 0 and
#                           #    abs(d - last_d) >=
#                           #    diameter + r + last_r):
#        self.print_state("Found bypass in positive: ",
#                         "shift_step=", shift_step,"d=", d)
#        break
#      last_angle = new_angle
#    positive_shift = shift
#    shift = 0
#    d = dst_to_follow
#    last_angle = angle
#    while 0 <= shift < pi:
#      new_angle, d = self.find_bypass(objects_around,
#                                         last_angle, d, -1)
#      shift_step = normalize_angle(new_angle - last_angle)
#      shift -= shift_step
#      self.print_state("Negative: shift=", shift, ", new_angle=",
#                       new_angle, ", last_angle=", last_angle,
#                       ", d=", d)
#      if shift_step > 0:
#        self.print_state("Bypass in negative is not found")
#        shift = pi + 1
#        break
#      elif shift_step == 0:# or (last_d != 0 and
#                           #    abs(d - last_d) >=
#                           #    diameter + r + last_r):
#        self.print_state("Found bypass in negative: ",
#                         "shift_step=", shift_step,"d=", d)
#        break
#      last_angle = new_angle
#    negative_shift = shift
#    if negative_shift >= pi and positive_shift >= pi:
#      # We stuck!
#      self.rule_dont_get_stuck(force = True)
#      return None
#    return normalize_angle(angle +
#      copysign(min(positive_shift, negative_shift),
#               negative_shift - positive_shift))
#
#  def find_bypass(self, objects_around, angle, dtf,
#                  bypass_direction):
#    min_dst = self.max_distance
#    result = angle
#    best_sum_radius = 0
#    for (obj, dst_to_edge, ang_to_obj, halfsector,
#         sum_radius) in objects_around:
#      if dst_to_edge > dtf:
#        continue
#      a_from_obj_to_target = normalize_angle(angle - ang_to_obj)
#      # ожидаемый угол поворота относительно угла к юниту
#      in_sector = abs(a_from_obj_to_target) <= halfsector
#      direction = copysign(1, a_from_obj_to_target)
#      #self.print_state("Found object: " +
#      #  str((obj.x, obj.y))) 
#      #self.print_state("dst=" + str(dst_to_edge) +
#      #  ", sum_radius=" + str(sum_radius) +
#      #  ", shift_from_center=" + str(a_from_obj_to_target) +
#      #  ", halfsector=" + str(halfsector) +
#      #  ", to_obj=" + str(ang_to_obj) +
#      #  ", r_direction=" + str(rotation_direction))
#      if in_sector and dst_to_edge < min_dst:
#        min_dst = dst_to_edge
#        best_sum_radius = sum_radius
#        #self.print_state("Found object on the way: " +
#        #  str((obj.x, obj.y)))
#        #self.print_state("dst=" + str(dst_to_edge) +
#        #  ", sum_radius=" + str(sum_radius) +
#        #  ", shift_from_center=" + str(a_from_obj_to_target) +
#        #  ", halfsector=" + str(halfsector) +
#        #  ", to_obj=" + str(ang_to_obj) +
#        #  ", old_r_direction=" + str(self.rotation_direction) +
#        #  ", r_direction=" + str(rotation_direction))
#        if (bypass_direction > 0 or
#            (direction > 0 and bypass_direction == 0)):
#          result = ang_to_obj + halfsector
#        else:
#          result = ang_to_obj - halfsector
#        from_edge = self.check_on_edge(obj, sum_radius)
#        if not (from_edge is None):
#          result = (angle_between(from_edge, angle) +
#                   bypass_direction * 2 * pi)
#          #angle = angle + bypass_direction * pi
#    return normalize_angle(result), min_dst + (best_sum_radius
#      - self.me.radius)
