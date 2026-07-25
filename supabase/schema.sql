-- KneeGuard AI — Supabase schema.
--
-- Run this once in your Supabase project: SQL Editor -> New query -> paste ->
-- Run. It is idempotent, so re-running it is safe.
--
-- Security model: the browser talks to Supabase directly with the *anon* key,
-- and Row Level Security is what stops one athlete reading another's results.
-- Every policy below is scoped to auth.uid(). Do not disable RLS, and never
-- put the service_role key in the frontend — it bypasses all of this.

-- ---------------------------------------------------------------------------
-- Assessments
-- ---------------------------------------------------------------------------

create table if not exists public.assessments (
    id                  uuid primary key default gen_random_uuid(),
    user_id             uuid not null default auth.uid()
                            references auth.users (id) on delete cascade,
    created_at          timestamptz not null default now(),

    -- What the athlete entered.
    age                 smallint,
    sex                 text,
    minutes_last_7d     integer,
    minutes_prior_28d   integer,
    consecutive_days    smallint,
    surface             text,
    soreness            smallint,
    sleep_hours         numeric(4, 1),
    prior_injury        boolean,
    contact_events      smallint,
    slide_tackles       smallint,
    acwr                numeric(6, 2),

    -- Headline results, as columns so history and trends can be queried
    -- without unpacking JSON.
    overall_index       numeric(5, 1),
    overall_band        text,
    primary_ligament    text,
    acl_index           numeric(5, 1),
    mcl_index           numeric(5, 1),
    pcl_index           numeric(5, 1),

    -- Full detail for re-displaying a past scorecard.
    --
    -- `scan` holds the numeric summary only — valgus angle, KASR, flexion,
    -- notes. No photo or video is ever stored: the app does not write uploaded
    -- media to disk and it must not end up in the database either. These are
    -- often minors, and a screening tool has no business keeping footage of
    -- them.
    ligaments           jsonb,
    action_plan         jsonb,
    scan                jsonb,
    explanation         jsonb
);

comment on table public.assessments is
    'One completed KneeGuard risk assessment. Never stores uploaded media.';

-- History is always read newest-first for one user.
create index if not exists assessments_user_created_idx
    on public.assessments (user_id, created_at desc);

alter table public.assessments enable row level security;

-- Policies are dropped first so the script can be re-run after edits.
drop policy if exists "Users read their own assessments"   on public.assessments;
drop policy if exists "Users insert their own assessments" on public.assessments;
drop policy if exists "Users delete their own assessments" on public.assessments;

create policy "Users read their own assessments"
    on public.assessments for select
    using (auth.uid() = user_id);

create policy "Users insert their own assessments"
    on public.assessments for insert
    with check (auth.uid() = user_id);

create policy "Users delete their own assessments"
    on public.assessments for delete
    using (auth.uid() = user_id);

-- Deliberately no UPDATE policy. A past assessment is a record of what was
-- measured at a point in time; editing it would make the history a lie.

-- ---------------------------------------------------------------------------
-- Profiles (display name, so the UI can greet someone by name)
-- ---------------------------------------------------------------------------

create table if not exists public.profiles (
    id           uuid primary key references auth.users (id) on delete cascade,
    display_name text,
    created_at   timestamptz not null default now()
);

alter table public.profiles enable row level security;

drop policy if exists "Users read their own profile"   on public.profiles;
drop policy if exists "Users insert their own profile" on public.profiles;
drop policy if exists "Users update their own profile" on public.profiles;

create policy "Users read their own profile"
    on public.profiles for select using (auth.uid() = id);

create policy "Users insert their own profile"
    on public.profiles for insert with check (auth.uid() = id);

create policy "Users update their own profile"
    on public.profiles for update using (auth.uid() = id);

-- Create the profile row automatically when someone signs up, so the app
-- never has to handle a missing profile.
create or replace function public.handle_new_user()
returns trigger
language plpgsql
security definer
set search_path = public
as $$
begin
    insert into public.profiles (id, display_name)
    values (new.id, split_part(new.email, '@', 1))
    on conflict (id) do nothing;
    return new;
end;
$$;

drop trigger if exists on_auth_user_created on auth.users;
create trigger on_auth_user_created
    after insert on auth.users
    for each row execute function public.handle_new_user();
