import torch
import torch.nn.functional as F
from torch import distributions as pyd
import utils
from agent.networks.actor import Actor
from agent.networks.critic import Critic


class Agent:
    def __init__(
        self,
        obs_shape,
        action_shape,
        device,
        lr,
        hidden_dim,
        critic_target_tau,
        num_expl_steps,
        update_every_steps,
        stddev_schedule,
        stddev_clip,
        use_tb,
    ):
        self.device = device
        self.lr = lr
        self.critic_target_tau = critic_target_tau
        self.update_every_steps = update_every_steps
        self.use_tb = use_tb
        self.num_expl_steps = num_expl_steps
        self.stddev_schedule = stddev_schedule
        self.stddev_clip = stddev_clip


        self.actor = Actor(obs_shape[0], action_shape, hidden_dim)

        self.critic = Critic(obs_shape[0], action_shape, hidden_dim)
        self.critic_target = Critic(obs_shape[0], action_shape, hidden_dim)
        self.critic_target.load_state_dict(self.critic.state_dict()) # load it with the critic's parameters

        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=lr)

        self.train() # trains actor & critic
        self.critic_target.train()

    def __repr__(self):
        return "rl"

    def train(self, training=True):
        self.training = training
        self.actor.train(training)
        self.critic.train(training)

    def act(self, obs, step, eval_mode):
        obs = torch.as_tensor(obs, device=self.device).float().unsqueeze(0)
        stddev = utils.schedule(self.stddev_schedule, step)
        dist = self.actor(obs.unsqueeze(0), stddev)
        
        if eval_mode:
            action = dist.sample()[0]
            # action = dist.mean[0]
            # print ("Eval mode ", action, "\n")
        else:
            # If step is less than the number of exploration steps, sample a random action.
            # Otherwise, sample an action from the distribution.
            if step < self.num_expl_steps:
                # action = dist.sample()[0]
                # action = pyd.uniform.Uniform(dist.low, dist.high).sample((action.shape[0], action.shape[1])) # how to get the right dimensionality here
                action = torch.FloatTensor(1, 2).uniform_(-1, 1).to(self.device)
            else:
                action = dist.sample()[0]

        return action.cpu().numpy()[0]

    def update_critic(self, obs, action, reward, discount, next_obs, step):
        metrics = dict()

        with torch.no_grad():
            stddev = utils.schedule(self.stddev_schedule, step)
            # TODO: Compute the target Q value
            # Hint: Use next obs and next action to compute the target Q value
            next_action = self.actor(next_obs.unsqueeze(0), stddev)
            next_action = next_action.sample()[0]
            target_Q = self.critic_target(next_obs, next_action) #next action 

        # TODO: Compute the Q value from the critic network
        Q = self.critic(obs, action)

        # TODO: Compute the critic loss
        critic_loss = torch.nn.functional.mse_loss(Q, reward + discount*target_Q.detach())

        # TODO: Optimize the critic network
        self.critic_opt.zero_grad()
        critic_loss.backward()
        self.critic_opt.step()




        if self.use_tb:
            metrics["critic_target_q"] = target_Q.mean().item()
            metrics["critic_loss"] = critic_loss.item()

        return metrics

    def update_actor(self, obs, step):
        metrics = dict()

        stddev = utils.schedule(self.stddev_schedule, step)

        # TODO: Get the action distribution from the actor network
        # and sample an action from the distribution
        dist = self.actor(obs.unsqueeze(0), stddev)
        action = dist.sample()[0]
        
        log_prob = dist.log_prob(action).sum(-1, keepdim=True) 

        # TODO: Get the Q value from the critic network
        Q = self.critic(obs, action)

        # TODO: Compute the actor loss
        # print("actor Q is ", Q, "\n")
        actor_loss = -torch.mean(Q)

        # TODO: Optimize the actor network
        self.actor_opt.zero_grad()
        actor_loss.backward()
        self.actor_opt.step()


        if self.use_tb:
            metrics["actor_loss"] = actor_loss.item()
            metrics["actor_logprob"] = log_prob.mean().item()
            metrics["actor_ent"] = dist.entropy().sum(dim=-1).mean().item()

        return metrics

    def update(self, replay_iter, step):
        metrics = dict()

        if step % self.update_every_steps != 0:
            return metrics

        batch = next(replay_iter) # collected from scratch
        obs, action, reward, discount, next_obs = utils.to_torch(batch, self.device)

        # convert to float
        obs = obs.float()
        next_obs = next_obs.float()
        action, reward, discount = action.float(), reward.float(), discount.float()

        if self.use_tb:
            metrics["batch_reward"] = reward.mean().item()

        # update critic -> first update critic using datasamples from the replay buffer
        metrics.update(
            self.update_critic(obs, action, reward, discount, next_obs, step)
        )

        # update actor -> then update actor; note that obs.detatch()
        metrics.update(self.update_actor(obs.detach(), step))

        # update critic target
        utils.soft_update_params(
            self.critic, self.critic_target, self.critic_target_tau
        )

        return metrics

    def save_snapshot(self):
        keys_to_save = ["actor", "critic"]
        payload = {k: self.__dict__[k].state_dict() for k in keys_to_save}
        return payload
